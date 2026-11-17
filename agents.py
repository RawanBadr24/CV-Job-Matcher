import json, re, time, os, requests
import pandas as pd
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # free; try Qwen2.5-3B-Instruct for better extraction

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")


# ---------------- LLM helpers ----------------
def ask_llm(prompt, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

stats = {"calls": 0, "failures": 0}      # used to measure reliability

def ask_json(prompt):
    """Ask the LLM for JSON. Returns a dict, or {} if the reply is not valid JSON."""
    stats["calls"] += 1
    try:
        return json.loads(re.search(r"\{.*\}", ask_llm(prompt), re.DOTALL).group())
    except Exception:
        stats["failures"] += 1
        return {}

def to_skills(value):
    """Turn the LLM's skills into a clean list of lowercase strings."""
    return [str(s).strip().lower() for s in value] if isinstance(value, list) else []

def same_skill(a, b):
    return a == b or (len(a) > 3 and len(b) > 3 and (a in b or b in a))

# ---- experience level: years of experience a job needs, from its title and description ----
TITLE_YEARS = {"intern": 0, "internship": 0, "trainee": 0, "graduate": 0, "entry": 0, "junior": 0, "jr": 0,
               "senior": 5, "sr": 5, "manager": 5, "lead": 6, "architect": 7, "principal": 8,
               "head": 8, "director": 8, "vp": 10, "chief": 10}
TOLERANCE = 1      # a job fits if it needs at most (your years + 1)

def to_years(value):
    """Turn the LLM's answer into a number of years (0 if unclear)."""
    try:
        return max(0.0, min(float(value), 50.0))
    except (TypeError, ValueError):
        found = re.search(r"\d+(\.\d+)?", str(value))
        return float(found.group()) if found else 0.0

def required_years(title, description):
    from_title = max([TITLE_YEARS.get(w, 0) for w in re.findall(r"[a-z]+", title.lower())], default=0)
    text = description.lower()
    found = re.findall(r"(\d{1,2})\s*\+?\s*(?:-\s*\d+\s*)?(?:years|yrs)\W+(?:of\s+)?(?:\w+\s+){0,3}experience", text)
    found += re.findall(r"experience\D{0,25}?(\d{1,2})\s*\+?\s*(?:-\s*\d+\s*)?(?:years|yrs)", text)
    from_text = min(int(n) for n in found) if found else 0
    return min(max(from_title, from_text), 15)


# ---------------- Agent 1: Retrieval (uploaded PDFs + jobs sheet + public API) ----------------
def load_jobs_sheet(sheet):
    """Read the jobs sheet (Google Sheet CSV link, CSV path or uploaded CSV)."""
    df = pd.read_csv(sheet).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"title", "company", "description"} - set(df.columns)
    if missing:
        raise ValueError(f"The jobs sheet is missing these columns: {', '.join(sorted(missing))}")
    link_col = next((c for c in ["link", "url", "job_link", "apply_link"] if c in df.columns), None)
    df["link"] = df[link_col].astype(str) if link_col else ""
    df["source"] = "sheet"
    return df[["title", "company", "description", "link", "source"]]


def retrieval_agent(cv_files, sheet, use_api, api_query):
    warnings, cvs = [], []
    for f in cv_files:
        name = os.path.basename(f.name)
        try:
            text = "\n".join(page.extract_text() or "" for page in PdfReader(f).pages)
        except Exception:
            warnings.append(f"{name}: could not be read as a PDF, skipped.")
            continue
        if not text.strip():
            warnings.append(f"{name}: no text found (scanned image?), skipped.")
            continue
        cvs.append({"file": name, "text": text})
    if not cvs:
        raise ValueError("None of the uploaded PDFs contained readable text.")

    jobs = []
    if sheet:
        jobs.append(load_jobs_sheet(sheet))
    if use_api:                                           # live jobs from a public API
        try:
            r = requests.get("https://remotive.com/api/remote-jobs",
                             params={"search": api_query, "limit": 5}, timeout=15).json()
            jobs.append(pd.DataFrame([{"title": j["title"], "company": j["company_name"],
                                       "link": j["url"], "source": "api",
                                       "description": re.sub(r"<[^>]+>", " ", j["description"])[:2000]}
                                      for j in r["jobs"]]))
        except Exception as e:
            warnings.append(f"Live jobs API unavailable: {e}")
    if not any(len(j) for j in jobs):
        raise ValueError("No jobs found. Add a jobs sheet or turn on the live jobs API.")
    return cvs, pd.concat(jobs, ignore_index=True), warnings


# ---------------- Agent 2: CV Parser (LLM) ----------------
def cv_parser_agent(cv):
    data = ask_json(f"""Extract information from this CV. Reply with JSON only, like:
{{"name": "Jane Doe", "skills": ["python", "sql"], "years_experience": 3}}
years_experience = years of full-time professional work only. Students, fresh graduates,
internships, courses and personal projects count as 0.

CV:
{cv['text'][:3000]}""")
    skills = to_skills(data.get("skills"))
    in_cv = sum(s in cv["text"].lower() for s in skills)           # skills that really appear in the CV
    return {"file": cv["file"], "name": str(data.get("name") or cv["file"]), "skills": skills,
            "years": to_years(data.get("years_experience", 0)),
            "grounded": round(100 * in_cv / len(skills)) if skills else 0}


# ---------------- Agent 3: Job Analyzer (LLM) ----------------
def job_analyzer_agent(job):
    data = ask_json(f"""Extract the required skills from this job post. Reply with JSON only, like:
{{"skills": ["python", "sql"]}}

Job title: {job['title']}
Description: {job['description'][:2000]}""")
    return {"title": job["title"], "company": job["company"], "source": job["source"],
            "link": str(job["link"]), "skills": to_skills(data.get("skills")),
            "years_needed": required_years(job["title"], job["description"])}


# ---------------- Agent 4: Matching (plain code, so the score is consistent) ----------------
def matching_agent(candidates, jobs):
    rows = []
    for c in candidates:
        for j in jobs:
            matched = [s for s in j["skills"] if any(same_skill(s, k) for k in c["skills"])]
            missing = [s for s in j["skills"] if s not in matched]
            score = round(100 * len(matched) / len(j["skills"])) if j["skills"] else 0
            fits = j["years_needed"] <= c["years"] + TOLERANCE          # experience level check
            rows.append({"file": c["file"], "candidate": c["name"], "candidate_years": round(c["years"], 1),
                         "job_title": j["title"], "company": j["company"], "source": j["source"],
                         "score": score, "fits": fits,
                         "experience_fit": "OK" if fits else f"Too senior (needs {j['years_needed']}+ yrs)",
                         "job_link": j["link"], "matched_skills": ", ".join(matched),
                         "missing_skills": ", ".join(missing)})
    df = pd.DataFrame(rows)
    # jobs that fit the experience level come first, then the best skill score
    df = df.sort_values(["file", "fits", "score"], ascending=[True, False, False]).reset_index(drop=True)
    df["rank"] = df.groupby("file").cumcount() + 1
    return df[["candidate", "candidate_years", "rank", "job_title", "company", "source", "score",
               "experience_fit", "job_link", "matched_skills", "missing_skills", "file"]]


# ---------------- Agent 5: Recommendation (LLM) ----------------
def recommendation_agent(candidate, top_matches):
    jobs = "\n".join(f"- {r.job_title} at {r.company}: {r.score}% skills match, experience: {r.experience_fit}, missing skills: {r.missing_skills or 'nothing'}"
                     for r in top_matches.itertuples())
    return ask_llm(f"""You are a career advisor.
Candidate: {candidate['name']} ({candidate['years']} years, skills: {', '.join(candidate['skills'])}).
Best job matches:
{jobs}

In 3 short sentences: say which job fits best and why, and which skills to learn to improve.
Only recommend jobs whose experience is OK. Mention a "Too senior" job only as a future goal.""", max_new_tokens=150)


# ---------------- Agent 6: Summary (LLM) ----------------
def summary_agent(matches):
    lines = "\n".join(f"- {r.candidate} ({r.candidate_years} years experience): best job {r.job_title} ({r.score}% skills, experience {r.experience_fit}), missing: {r.missing_skills or 'nothing'}"
                      for r in matches[matches["rank"] == 1].itertuples())
    return ask_llm(f"""Summarize these CV-to-job matching results in 4 short sentences for a hiring manager:
who fits best, and which skill gaps appear most often.

{lines}""", max_new_tokens=200)


# ---------------- Automated actions ----------------
def save_to_sheet(matches, path="results.csv"):       # CSV opens in Excel / Google Sheets
    matches.drop(columns="file").to_csv(path, index=False)


def generate_pdf_report(matches, summary, path="report.pdf"):
    styles = getSampleStyleSheet()
    story = [Paragraph("CV - Job Matching Report", styles["Title"]),
             Paragraph("<b>Summary:</b> " + escape(summary), styles["Normal"]), Spacer(1, 15)]
    for _, group in matches.groupby("file"):
        story.append(Paragraph(escape(group.candidate.iloc[0]), styles["Heading2"]))
        story.append(Paragraph("<b>AI advice:</b> " + escape(group.advice.iloc[0]), styles["Normal"]))
        for r in group.head(3).itertuples():
            line = escape(f"{r.rank}. {r.job_title} at {r.company} - {r.score}% skills, experience: {r.experience_fit} (missing: {r.missing_skills or 'none'})")
            if r.job_link:
                href = escape(r.job_link).replace('"', "&quot;")
                line += f' - <link href="{href}" color="blue">job link</link>'
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 15))
    SimpleDocTemplate(path).build(story)


# ---------------- Live evaluation (works on any uploaded CVs, no answer key needed) ----------------
def live_evaluation(candidates, matches, seconds):
    rows = []
    for c in candidates:
        best = int(matches[matches.file == c["file"]]["score"].max())
        ok = len(c["skills"]) >= 3 and c["grounded"] >= 80
        top3 = matches[(matches.file == c["file"]) & (matches["rank"] <= 3)]
        rows.append({"file": c["file"], "name": c["name"], "skills_found": len(c["skills"]),
                     "skills_really_in_cv_%": c["grounded"], "experience_years": round(c["years"], 1),
                     "top3_fit_experience": "yes" if (top3["experience_fit"] == "OK").all() else "no",
                     "best_match_%": best, "status": "OK" if ok else "CHECK"})
    valid = stats["calls"] - stats["failures"]
    return {"per_cv": pd.DataFrame(rows), "json_valid": f"{valid}/{stats['calls']}",
            "seconds": seconds, "seconds_per_cv": round(seconds / len(candidates), 1),
            "outputs": {f: os.path.exists(f) for f in ["results.csv", "report.pdf"]}}


# ---------------- Orchestrator ----------------
def run_pipeline(cv_files, sheet=None, use_api=True, api_query="data", years_override=None):
    if not cv_files:
        raise ValueError("Please upload at least one CV (PDF).")
    stats["calls"] = stats["failures"] = 0
    start = time.time()

    cvs, jobs, warnings = retrieval_agent(cv_files, sheet, use_api, api_query)        # 1. get the data
    candidates = [cv_parser_agent(cv) for cv in cvs]                                  # 2. understand CVs
    if years_override is not None:                                                    # user corrected the experience
        for c in candidates:
            c["years"] = float(years_override)
    job_list = [job_analyzer_agent(job) for job in jobs.to_dict("records")]           # 3. understand jobs
    matches = matching_agent(candidates, job_list)                                    # 4. score + rank

    advice = {c["file"]: recommendation_agent(c, matches[matches.file == c["file"]].head(3))
              for c in candidates}                                                    # 5. advice
    matches["advice"] = matches["file"].map(advice)
    summary = summary_agent(matches)                                                  # 6. overall summary

    save_to_sheet(matches)                                                            # action: sheet
    generate_pdf_report(matches, summary)                                             # action: report
    seconds = round(time.time() - start, 1)

    return matches, summary, live_evaluation(candidates, matches, seconds), warnings

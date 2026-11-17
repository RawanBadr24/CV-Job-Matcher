import streamlit as st

st.set_page_config(page_title="CV - Job Matcher", page_icon="🎯", layout="wide")
st.title("🎯 CV ↔ Job Matcher (Multi-Agent)")

with st.spinner("Loading the AI model (first time only)..."):
    import agents

# ---------------- Sidebar: inputs ----------------
with st.sidebar:
    st.header("1. Upload CVs")
    cv_files = st.file_uploader("CV PDFs (any number)", type="pdf", accept_multiple_files=True)

    st.header("2. Jobs")
    sheet_url = st.text_input("Google Sheet CSV link (optional)")
    sheet_file = st.file_uploader("...or upload a jobs CSV", type="csv")
    st.caption("Sheet columns: title, company, description. Optional: link (job page or application URL).")
    use_api = st.checkbox("Also fetch live jobs (Remotive API)", value=True)
    api_query = st.text_input("Live jobs keyword", "data")

    st.header("3. Your experience")
    levels = {"Auto-detect from the CV": None, "Fresh graduate (0 years)": 0, "Junior (1-2 years)": 2,
              "Mid-level (3-5 years)": 4, "Senior (6+ years)": 7}
    level = st.selectbox("Experience level", list(levels))
    st.caption("Use this if the detected experience is wrong. It applies to all uploaded CVs.")

    run = st.button("Run matching", type="primary")

if run:
    try:
        with st.spinner("The agents are working... this can take a few minutes."):
            sheet = sheet_file or sheet_url.strip() or None
            st.session_state["result"] = agents.run_pipeline(cv_files, sheet, use_api, api_query, levels[level])
    except Exception as e:
        st.error(str(e))

if "result" not in st.session_state:
    st.info("Upload your CV PDFs and choose a jobs source in the sidebar, then click **Run matching**.")
    st.stop()

matches, summary, evaluation, warnings = st.session_state["result"]
for w in warnings:
    st.warning(w)

tab_results, tab_eval = st.tabs(["Results", "Evaluation"])

# ---------------- Results tab ----------------
with tab_results:
    st.info(summary)
    candidate = st.selectbox("Candidate", sorted(matches["candidate"].unique()))
    top_n = st.slider("Number of top jobs", 1, 10, 5)
    hide_senior = st.checkbox("Hide jobs that are too senior for this CV", value=True)

    cand = matches[matches["candidate"] == candidate].sort_values("rank")
    st.caption(f"Detected experience: {cand['candidate_years'].iloc[0]} years. Jobs that need much more "
               "experience are ranked last and marked 'Too senior'.")
    df = cand[cand["experience_fit"] == "OK"] if hide_senior else cand
    df = df.head(top_n)

    if df.empty:
        st.warning("No job fits this CV's experience level. Untick the box above to see all jobs.")
    else:
        st.bar_chart(df.set_index("job_title")["score"])
        st.dataframe(df[["rank", "job_title", "company", "source", "score", "experience_fit", "job_link",
                         "matched_skills", "missing_skills"]],
                     hide_index=True, column_config={"job_link": st.column_config.LinkColumn("Job link")})
    st.markdown("**AI advice:** " + cand["advice"].iloc[0])

    c1, c2 = st.columns(2)
    c1.download_button("Download results sheet (CSV)", matches.drop(columns="file").to_csv(index=False),
                       "results.csv", "text/csv")
    c2.download_button("Download PDF report", open("report.pdf", "rb").read(), "report.pdf", "application/pdf")

# ---------------- Evaluation tab ----------------
with tab_eval:
    st.subheader("Live evaluation of this run")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid LLM JSON", evaluation["json_valid"])
    c2.metric("Total time", f"{evaluation['seconds']}s")
    c3.metric("Time per CV", f"{evaluation['seconds_per_cv']}s")
    c4.metric("Outputs created", f"{sum(evaluation['outputs'].values())}/{len(evaluation['outputs'])}")
    st.dataframe(evaluation["per_cv"], hide_index=True)
    st.caption("OK = at least 3 skills were extracted AND at least 80% of them literally appear in the CV text "
               "(so the model did not invent them). CHECK = review that CV manually.")

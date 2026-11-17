# Multi-Agent CV ↔ Job Matcher

A multi-agent AI system that reads CV PDFs, matches them against job posts, and automatically produces a ranked results sheet (with job links), a PDF report and a live dashboard. Built with free open models (Qwen2.5) and Streamlit.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/cv-job-matcher/blob/main/CV_Job_Matcher_Multi_Agent.ipynb)

## Features
- **Upload any number of CV PDFs** in a Streamlit web app.
- **Three data sources:** uploaded CVs (PDF), a jobs sheet (Google Sheet CSV link or CSV upload) and live jobs from a public API (Remotive).
- **Experience-level filter:** jobs that need much more experience than the CV has (e.g. "Senior", "Head of") are ranked last and marked *Too senior*, so fresh graduates are not matched to senior roles.
- **Automated outputs:** results sheet (`results.csv`) with job links, a PDF report, and the dashboard itself.
- **Evaluation:** a labeled test set (accuracy, valid-JSON rate, speed, error handling) plus a *live evaluation* tab that checks any uploaded CV, including whether the extracted skills really appear in the CV text.

## How it works

```
 Uploaded CV PDFs ────┐
 Jobs sheet (link/CSV)┼─► 1. Retrieval Agent
 Live jobs API ───────┘            │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            2. CV Parser Agent (LLM)   3. Job Analyzer Agent (LLM)
                     └─────────────┬─────────────┘
                                   ▼
                        4. Matching Agent (code)
                                   ▼
                      5. Recommendation Agent (LLM)
                                   ▼
                        6. Summary Agent (LLM)
                                   ▼
        Actions: results sheet (with job links) · PDF report · live dashboard
```

| Agent | Type | Job |
|---|---|---|
| Retrieval | code | Reads CV PDFs, the jobs sheet and the live API |
| CV Parser | LLM | Extracts name, skills and years of experience |
| Job Analyzer | LLM + rules | Extracts required skills; reads required years from the title and description |
| Matching | code | Scores skill overlap, applies the experience filter, ranks jobs |
| Recommendation | LLM | Writes short career advice per candidate |
| Summary | LLM | Summarizes all results for a hiring manager |

Matching is plain code on purpose: a small LLM is unreliable at scoring, so the LLMs *understand* text and code does the *scoring*, which keeps rankings consistent and explainable.

## Run it

**Google Colab (recommended, free GPU):** click the Colab badge above, switch to a GPU runtime, run all cells, and open the public link printed by the last cell.

**Locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the Qwen model. A GPU is strongly recommended.

## Jobs sheet format
CSV or Google Sheet (published as CSV) with the columns `title`, `company`, `description`, and optionally `link` (the job page or application URL). See `sample_data/jobs_example.csv`.

## Limitations
- A small LLM (Qwen2.5-1.5B) can miss or rename skills. A larger model improves extraction.
- Scanned (image-only) PDFs have no text and are skipped with a warning.
- Matching uses skills and experience level only (not location, salary or education).
- Required experience is read from the job title and description with simple rules; a job that never mentions seniority is treated as open to everyone.

## Project structure
```
├── CV_Job_Matcher_Multi_Agent.ipynb   # full notebook: agents, app, tests, launch
├── agents.py                          # agents, actions, evaluation, pipeline
├── app.py                             # Streamlit interface
├── sample_data/jobs_example.csv       # example jobs sheet
└── requirements.txt
```

## made by
Rawan Badr El din Kamal

# Ai_ATS_ResumeChecker


A web app that scores how well a resume matches a job description and returns actionable feedback. Built with FastAPI + Streamlit, using spaCy and Sentence Transformers for NLP and the Groq API for LLM-generated suggestions.

## What it does

1. Upload a resume (PDF / DOC / DOCX) and paste a job description.
2. The backend parses the resume, extracts skills and experience, and compares them to the JD using semantic similarity.
3. You get an ATS score, a breakdown by category (formatting, keywords, content, skill validation, ATS compatibility), and LLM-written suggestions for what to improve.
4. Past analyses are saved to your account so you can revisit them.

## Tech stack

- **Frontend:** Streamlit
- **Backend:** FastAPI (Python)
- **NLP:** spaCy (`en_core_web_md`), Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM:** Groq API (Llama 3)
- **Auth + Database:** Supabase (email/password and Google OAuth)
- **PDF report export:** WeasyPrint + Jinja2

## Project structure

```
ATS_SCORER/
├── backend/              FastAPI app, NLP services, API routes
├── frontend/             Streamlit app, views, components
├── jupyter notebooks/    Research and dataset prep (not used at runtime)
├── ml model/             Exported ML artifacts
├── requirements.txt      Combined backend + frontend dependencies
└── .env.example          Template for environment variables
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ATS_SCORER
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
```

Activate this virtual environment in each terminal before running the backend
or frontend so both use the installed project dependencies.

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

WeasyPrint needs system libraries on Linux:

```bash
# Fedora
sudo dnf install -y cairo pango gdk-pixbuf2 libffi

# Debian / Ubuntu
sudo apt install -y libcairo2 libpango-1.0-0 libpangoft2-1.0-0 libffi-dev
```

### 3. Configure environment variables

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Configure `SUPABASE_URL`, `SUPABASE_KEY` (service role), `SUPABASE_JWT_SECRET`,
and `GROQ_API_KEY` for the backend. The Streamlit frontend only needs the
Supabase URL and the public anon key. Keep the service-role key, JWT secret,
and Groq key out of Streamlit secrets and source control.

For local development, create `.streamlit/secrets.toml` at the repository root:

```toml
[supabase]
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-public-anon-key"

[backend]
url = "http://127.0.0.1:8000"
```

Alternatively, set `SUPABASE_URL` and `SUPABASE_ANON_KEY` in the frontend's
environment. (Optional) Configure Google OAuth in the Supabase dashboard.

### 4. Run the backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now at `http://localhost:8000`.

### 5. Run the frontend

In a new terminal (with the venv activated):

```bash
python -m streamlit run frontend/streamlit_app.py
```

The app opens at `http://localhost:8501`.

### Backend URL configuration

The frontend defaults to `http://127.0.0.1:8000`, which works when both apps
run on the same machine. The backend must be running separately; open
`http://127.0.0.1:8000/api/v1/health` to confirm it is ready.

If the frontend is deployed (for example, on Streamlit Community Cloud),
`localhost` points to the frontend container, not your backend. Deploy the
backend separately at a publicly reachable HTTPS URL, then add its API base
URL to the app's Streamlit Cloud secrets (or set `BACKEND_URL` in the frontend
environment):

```toml
[backend]
url = "https://your-backend.example.com"
```

This is the backend's public base URL, not a Supabase URL. The Streamlit app
uses it to call the backend API; it does not expose backend credentials.

Alternatively, set the `BACKEND_URL` environment variable on the frontend
deployment. If running the backend with `python -m backend.main`, its listen
address and port can be set with `HOST` and `PORT` (defaults: `0.0.0.0` and
`8000`).

## Notes for students

- **Never commit `.env` or `secrets.toml`** — they hold API keys. Both are in `.gitignore`; check before you push.
- The first run downloads the Sentence Transformer model (~80 MB). It's cached afterwards.
- If Groq is unavailable or its API key is invalid, resume/JD parsing falls back to local text extraction. The analysis still runs, but the extracted details may be less accurate.
- `jupyter notebooks/` and `ml model/` are for experimentation and aren't required to run the app.
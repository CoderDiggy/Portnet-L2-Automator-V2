# Portnet-L2-Automator-V2

Minimal instructions to run this project.

Requirements
- Python 3.11 (required)
- pip

What it uses
- FastAPI — web framework to build the API and web UI.
- Uvicorn — ASGI server for development and simple production setups.
- Gunicorn + Uvicorn workers — recommended for production.
- SQLAlchemy — ORM for database models.
- Pydantic — data validation and serialization for API payloads.
- Jinja2 — server-side templating for the web UI.
- MySQL (or other SQL DB) — persistence backend (use a SQLAlchemy-compatible driver such as PyMySQL).
- Azure OpenAI — AI incident analysis and generation (configured via environment variables).
- Alembic — optional DB migrations for production.
- httpx / requests — HTTP client utilities used by services.
- python-dotenv — optional, for loading .env files in development.
- pytest / pytest-asyncio — testing tools.
- Bootstrap 5 — frontend styles and responsiveness.
- Docker — containerization for deployment (Dockerfile included).

Quick setup
1. Clone and enter project:
   git clone https://github.com/CoderDiggy/Portnet-L2-Automator-V2.git
   cd Portnet-L2-Automator-V2

2. Create and activate a Python 3.11 virtualenv:
   python3.11 -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt

4. Copy and edit environment file (fill in DB and Azure OpenAI values if needed):
   cp .env.example .env

Run (quick test)
- Change into the application subfolder that contains simple_main.py:
  cd "AI Assistant Python" (IMPORTANT)

- Start the app:
  python simple_main.py
  # or explicitly:
  python3.11 simple_main.py

Access
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

That's it.

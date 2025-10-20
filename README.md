# Portnet-L2-Automator-V2

A FastAPI-based assistant for L2 automation and incident analysis. This repository contains a Python FastAPI implementation with services for AI analysis, a knowledge base, and training-data management.

Important: Please use Python 3.11 for development and production. The project has been developed and tested with Python 3.11 and dependency compatibility assumes that version.

## Contents

- app/ — FastAPI application code (models, services, templates, static)
- requirements.txt — Python dependencies
- .env.example — Example environment variables
- Dockerfile — Example Dockerfile for deployment

## Prerequisites

- Python 3.11 (required)
- pip
- MySQL (or other DB if configured)
- Optional: Docker & Docker Compose for containerized deployment
- Azure OpenAI credentials if using AI features

Verify Python version:

python --version
# Expect: Python 3.11.x

## Quick Start (Development)

1. Clone the repository:

git clone https://github.com/CoderDiggy/Portnet-L2-Automator-V2.git
cd Portnet-L2-Automator-V2

2. Create and activate a virtual environment using Python 3.11:

python3.11 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

3. Install dependencies:

pip install -r requirements.txt

4. Copy environment file and configure:

cp .env.example .env
# Then edit .env and fill in values for:
# DATABASE_URL, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
# AZURE_OPENAI_DEPLOYMENT_ID, AZURE_OPENAI_API_VERSION, etc.

5. Initialize or migrate the database:
- For development you can let SQLAlchemy create tables automatically on first run.
- For production, use Alembic for migrations:

pip install alembic
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

6. Run the app (development):

# Simple runner (use this for quick testing)
python simple_main.py

# If you need to ensure Python 3.11 explicitly:
python3.11 simple_main.py

# Option: use Uvicorn (if the ASGI app is exposed in app.main)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Open:
- Web UI: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Running in Production

Use Uvicorn with a process manager or Gunicorn with Uvicorn workers.

Example with Gunicorn + Uvicorn workers:

pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

Set environment variables securely in your environment (do not commit secrets).

Recommended production steps:
- Use a reverse proxy (nginx) for TLS termination and static files.
- Configure logging, monitoring, and process supervision (systemd/container orchestrator).
- Run database migrations with Alembic prior to starting the app.

## Docker

Example Dockerfile (already included in the repo as reference):

FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Build and run:

docker build -t portnet-l2-automator-v2 .
docker run -e "DATABASE_URL=..." -e "AZURE_OPENAI_API_KEY=..." -p 8000:8000 portnet-l2-automator-v2

## Configuration

All runtime configuration is read from environment variables (see `.env.example`). Key variables:

- DATABASE_URL — SQLAlchemy-compatible DB connection string (e.g., mysql+pymysql://user:pass@host:3306/dbname)
- AZURE_OPENAI_API_KEY — Azure OpenAI API key
- AZURE_OPENAI_ENDPOINT — Azure OpenAI endpoint
- AZURE_OPENAI_DEPLOYMENT_ID — Model deployment ID (recommended: gpt-4 if available)
- AZURE_OPENAI_API_VERSION — API version
- Additional app-specific config may exist in config files or environment variables; check `app` package for details.

## Usage Examples

- Web UI: Use the home page to submit incidents for analysis.
- API:
  - GET /api/training-data — list training data
  - POST /api/training-data — create training data
  - GET /api/knowledge — list knowledge entries
  - POST /api/knowledge — create knowledge entry
  - POST /api/knowledge/import-word — import knowledge from Word document

Refer to the interactive Swagger documentation at /docs for full request/response formats.

## Development Guidelines

- Code structure:
  - app/models — SQLAlchemy models
  - app/schemas — Pydantic schemas
  - app/services — AI, knowledge-base, training-data services
  - app/templates — Jinja2 templates for the UI
  - app/static — CSS/JS/assets
  - app/main.py — FastAPI app and routers

- Adding features:
  1. Add models (SQLAlchemy) and schemas (Pydantic)
  2. Add services under app/services
  3. Add routes in main.py or separate routers and include them
  4. Add templates and static files as needed

- Testing:

pip install pytest pytest-asyncio
pytest

## Troubleshooting

- "Cannot connect to database": verify DATABASE_URL, network access, and DB user permissions.
- "Missing Azure OpenAI credentials": ensure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set.
- "Model errors or rate limits": confirm deployment ID, model availability, and subscription quotas.
- Dependency issues: ensure Python 3.11 is active in your virtual environment and reinstall dependencies.

If you encounter errors, consult application logs (stdout, configured log files) for stack traces and more detail.

## Security & Secrets

- Never commit `.env` or secrets to the repository.
- Use environment-specific secret management in production (e.g., Vault, cloud secret managers).
- Use HTTPS/TLS in production.

## Contributing

Contributions are welcome. Please open issues or pull requests describing the change. Follow repository coding standards and include tests for new functionality.

## License

This project follows the same license as the original C# version. See the LICENSE file (if present) for details.

## Contact

For questions about this repository, open an issue or contact the maintainer.

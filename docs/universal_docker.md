# Universal Docker Development Strategy

## Goals

- Consistent development environments across OS/hardware
- One-command onboarding for new contributors
- Production-like local stack (app + services)
- Automated quality and security validation
- Enterprise-ready CI/CD hooks

## Architecture

- Dockerfile: GPU-ready base with Python 3.11, in-container venv
- docker compose: `app` + optional `postgres` + `redis`
- Dev volumes: bind-mount repo to `/workspace` for hot reload
- Ports: 8888 (notebooks), 8501 (Streamlit), 8000 (FastAPI)

## One-command setup

- Copy `.env.example` to `.env` and fill values (optional for W&B/MLflow)
- Start stack:
  - `docker compose up -d --build`
- Enter app container shell:
  - `docker compose exec app bash`

## Workflows

- Tests: `pytest -q`
- Lint/format: `pre-commit run --all-files`
- Notebooks: `jupyter lab --ip=0.0.0.0 --no-browser --allow-root`
- Streamlit demo: `streamlit run src/eegintent/realtime/app.py`

## Security & Quality

- CI runs ruff/black/mypy/pytest
- Optional: Trivy image scan and Hadolint for Dockerfile (see CI section)

## Production parity

- Postgres and Redis services included for pipelines that need persistence or queues
- Environment variables are passed via `.env`

## Troubleshooting

- If GPUs aren’t detected: ensure `--gpus all` is available (NVIDIA Container Toolkit). On compose, we request GPU via device reservations.
- For permissions on bind mounts, align UID/GID or run as root in dev containers.

.PHONY: help up down logs exec scan test lint fmt precommit notebooks

help:
	@echo "Targets: up, down, logs, exec, scan, test, lint, fmt, precommit, notebooks"

up:
	bash scripts/dev-up.sh

down:
	bash scripts/dev-down.sh

logs:
	bash scripts/dev-logs.sh

exec:
	bash scripts/dev-exec.sh

scan:
	bash scripts/dev-scan.sh

test:
	docker compose exec app pytest -q || true

lint:
	docker compose exec app ruff check . && docker compose exec app mypy || true

fmt:
	docker compose exec app ruff format . && docker compose exec app black . || true

precommit:
	docker compose exec app pre-commit run --all-files || true

notebooks:
	docker compose exec -d app jupyter lab --ip=0.0.0.0 --no-browser --allow-root

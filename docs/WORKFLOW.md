# Development Workflow

## Branching strategy

- `main` is stable and protected.
- Use feature branches: `feat/...`, `fix/...`, `docs/...`.
- Rebase or squash merge via PR after review.

## CI/CD

- Build: lint (ruff), format check (black), type check (mypy).
- Test: pytest minimal install to avoid heavy deps; full GPU tests optional.
- Deploy: tag releases `vX.Y.Z` to publish Docker image to GHCR.

## Reviews

- Small PRs (<500 lines) preferred.
- Include tests and docs updates.
- Ensure reproducibility (seed, versions in artifacts).

## Releases

- Keep `CHANGELOG.md` updated.
- Use semantic versioning.

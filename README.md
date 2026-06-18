# `badges` branch — generated artifacts only

This is an **orphan branch** (no shared history with `main`) that holds only the
auto-generated milestone-badge JSON files. It is **not** meant to track `main`;
GitHub showing it as "behind" is expected and harmless.

Each file is a [shields.io endpoint](https://shields.io/endpoint) payload written
by the CI workflow (`.github/workflows/ci.yml`) on every push:

- `main.json` — milestone progress on `main` (the starter branch: `0/8`).
- `solutions.json` — milestone progress on `solutions` (the reference: `8/8`).

The READMEs on those branches point their **Milestones** badge at the matching
file here. Don't edit these by hand — CI overwrites them.

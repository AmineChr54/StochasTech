# StochasTech — Documentation

StochasTech is a research + engineering project that builds a **differentiable Heston-model calibrator** and benchmarks its Value-at-Risk (VaR) forecasts against plain Geometric Brownian Motion (GBM) on real equity data. The Phase 1 deliverable is one polished Python repo + a short paper. C++/CUDA acceleration and ORIA integration are deferred to later phases.

## How to read this folder

| Doc | Purpose |
|-----|---------|
| [01_overview.md](01_overview.md) | What the project is and what it produces. |
| [02_roadmap.md](02_roadmap.md) | The 6-week sprint plan, week by week. |
| [03_scope.md](03_scope.md) | What is in scope for v1, what is explicitly out, why. |
| [math/](math/) | Mathematical foundations. One file per concept, full LaTeX derivations. **Each math file is paired with a code module — code merges only when the math doc exists.** |
| [methods/](methods/) | ML methods survey. `neural_sde.md` = the chosen approach. `future_work.md` = parking lot. |
| [architecture/](architecture/) | Repo layout + tech stack. `phase2_cpp.md` is the deferred C++/CUDA spec. |
| [reference/](reference/) | External inspirations and reading list. ORIA writeup lives here as motivation. |
| [archive/](archive/) | Earlier drafts kept for provenance. Not authoritative. |

## Conventions

- **Math docs are mandatory.** Every file in `stochastech/sde/` and `stochastech/calibration/` ships with a companion `doc/math/*.md` containing the derivation. See [math/00_index.md](math/00_index.md) for the template.
- LaTeX uses GitHub-native `$...$` and `$$...$$`. Define every symbol the first time it appears.
- Cross-doc links are relative.
- The single source of truth for scheduling is [02_roadmap.md](02_roadmap.md). The single source of truth for what the project *is* is [01_overview.md](01_overview.md). Anything in `archive/` that contradicts those two is wrong.

## Project status

Phase 1 (6-week solo sprint) — not yet started. See [02_roadmap.md](02_roadmap.md) for the week-by-week schedule.

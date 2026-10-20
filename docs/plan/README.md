# NeuroSight AI — Plan Set

Six-document build system (PRD → Technical Design → App Flow → UI/UX Brief → Backend Design → Engineering Plan), produced 2026-09-14, with a review of the superseded upgrade proposal.

| # | Document | Read it for |
|---|----------|-------------|
| 00 | [PLAN-REVIEW.md](00-PLAN-REVIEW.md) | Why `neurosight_new_plan.md` is not executable as written (3 fatal, 5 scientific, 12 engineering defects) and what changed |
| 01 | [01-PRD.md](01-PRD.md) | Users, scope, functional/non-functional requirements, honest success criteria, risks |
| 02 | [02-TECHNICAL-DESIGN.md](02-TECHNICAL-DESIGN.md) | Architecture, decisions & trade-offs, package layout, training/serving protocols, constraints |
| 03 | [03-APP-FLOW.md](03-APP-FLOW.md) | Routes, wizard journeys, every UI state and action |
| 04 | [04-UI-UX-BRIEF.md](04-UI-UX-BRIEF.md) | Design tokens, components, interaction rules, accessibility (WCAG AA) |
| 05 | [05-BACKEND-DESIGN.md](05-BACKEND-DESIGN.md) | API surface, data model, jobs, security/PHI posture, inference runtime |
| 06 | [06-ENGINEERING-PLAN.md](06-ENGINEERING-PLAN.md) | 30 tasks in dependency order with tests, done criteria, and wall-clock estimates |

**Execution order:** read 00 → 01 → 02, then implement via 06 (docs 03–05 are referenced by tasks).
**First action:** `T1.1` — acquire the missing demographics CSV (`oasis_longitudinal_demographics.csv` / Kaggle `oasis_longitudinal.csv`); nothing downstream runs without it.

# GPT Design Critique of Fine-Tuning Docs

> Canonical Header
- Version: 0.1.0
- Status: See STATUS.md
- Owners: Architecture TL; ML Lead; Docs Owner
- Last Updated: 2025-09-16
- Linked Commit: 682ba289170b (describe: 682ba28)

## Document Scope
- Critical review of design docs with prioritized, actionable recommendations.
- Not authoritative for status or counts; see STATUS.md for metrics.

## Out of Scope / Planned
- Implementations and code changes (tracked in repository and future PRs).
- Detailed benchmarking methodology (to be added as a separate doc).

Last Reviewed: 2025-09-16

Owners Suggested: Architecture (Tech Lead), MLX Backend (ML Lead), Data/Config (Data Lead), CLI/API/UI (Product Eng)

## Executive Summary

The design set is ambitious, well organized, and generally consistent with the current codebase and tests. Strengths include a clear MLX-first vision, solid modular separation (models/data/training/config), and highly actionable CLI-first workflows. The largest issues are: (1) status/test-count inconsistency across documents, (2) scope inflation where docs promise features not present in code (DB, API/UI breadth, format support), (3) missing engineering rigor for performance claims and memory math, and (4) lack of security, privacy, and reliability SLOs. Addressing these will make the docs a reliable source of truth and reduce future rework.

## Highlights (What’s Working Well)

- Clear layering and boundaries: backends, models, training, data, config.
- Strong MLX narrative with fallback to PyTorch; good practical guidance.
- CLI-centric UX with concrete commands promotes reproducibility.
- Data templates and validation described concretely and reflected in tests.
- Roadmap phased by capability with useful user outcomes.

## Gaps, Inconsistencies, and Risks

1) Cross-document status/test-count drift
- ARCHITECTURE: “PHASE 2 COMPLETE (290+ tests)”
- MLX_ARCHITECTURE: “Phase 1 Complete; Phase 2 LoRA Started”
- MLX_INTEGRATION: “Phase 1 Complete; PHASE 2 COMPLETE (290+ tests)”
- PROJECT_STRUCTURE: “Phase 1 Complete; Phase 2 Week 1-2 Complete (94 new tests)”
- TECH_STACK: “PHASE 2 COMPLETE (290+ tests)”
Impact: Erodes trust in docs as a source of truth.

2) Scope overstatement vs code
- Storage: SQLite, HDF5, Alembic listed, but repo shows no active DB usage or migrations.
- Data formats: Parquet/CSV/TSV/HTML mentioned; code/tests show JSON/JSONL only.
- Web/API: Rich FastAPI endpoints and Streamlit dashboard described; code has placeholders but not production-ready.
- Quantization: Bitsandbytes/QLoRA listed; actual quantization support appears incomplete in code.
Impact: Reader may assume features exist and design dependencies are satisfied.

3) Performance and memory claims lack methodology
- Tokens/sec numbers and power efficiency assertions lack hardware specs, dataset, model variants, sequence length, precision, and measurement protocol.
- Memory multiplier approximations (≈5×) are generic and do not distinguish optimizer states, activation checkpointing, mixed precision, or LoRA-only training deltas.
Impact: Hard to reproduce; may lead to mis-set expectations and regressions.

4) Security, privacy, and secrets handling are under-specified
- `passwords.yml` is mentioned, but no threat model, rotation guidance, or macOS Keychain guidance.
- No policy for model/dataset PII handling, contamination detection thresholds, or audit logging.
Impact: Risk for users handling sensitive datasets; unclear compliance posture.

5) Reliability/SLOs and failure handling
- No availability/error budget goals for CLI, API, training jobs.
- No restartability guarantees or checkpoint frequency SLOs.
- No reproducibility guarantees (random seeds, determinism policy) and artifact lineage.
Impact: Hard to reason about operational reliability and debugging.

6) Checkpoint portability and compatibility
- Good intent to save universal checkpoints, but no formal spec (e.g., naming, dtype policy, param mapping tables, versioning, schema evolution guarantees).
Impact: Future incompatibilities and migration churn.

7) Architectural diagrams and interface contracts
- ASCII diagrams help, but lack C4-context/container diagrams and concrete interface schemas (CLI args schema, API OpenAPI schemas, dataset/templating schemas, config JSONSchema).
Impact: Harder onboarding and review; ambiguity at component boundaries.

8) Packaging/distribution plan lacks trade-off analysis
- Homebrew, PyPI, and DMG are all proposed without constraints, signing pathways, or update channels defined.
Impact: Duplicated effort and user fragmentation.

## Concrete Recommendations

### A. Make docs a single source of truth
- Add a canonical header to every design doc: Version, Status, Owners, Last Updated, Linked commit.
- Centralize status/tests into `docs/design/STATUS.md` (CI-generated) and link from all docs.
- Add “Document Scope” and “Out of Scope/Planned” sections to each doc.

### B. Align scope with the codebase
- Storage: Move SQLite/Alembic/HDF5 from current scope to “Planned,” or add a minimal MVP (e.g., a single `runs.db` with migrations) and document schema.
- Data formats: Narrow in docs to JSON/JSONL now; mark CSV/TSV/Parquet as Phase 3 with an acceptance checklist.
- Web/API/UI: Mark as Phase 4 with explicit MVP endpoints and pages; include OpenAPI stub and page list with owners.
- Quantization: Document current state explicitly; list missing functionality and ETA.

### C. Add performance and memory methodology
- Create `docs/perf/METHODOLOGY.md` describing: hardware (chip, RAM), OS, Python, MLX/PyTorch versions, model, tokenizer, sequence length, precision, batch sizes, dataset, warmup, measurement duration, metrics (tokens/sec, step time p50/p90), and profiling tools.
- Add a memory model note distinguishing: weights, grads, optimizer states, activations, buffers, LoRA-only updates; provide formulas per optimizer and dtype.
- Publish a small benchmark harness in `scripts/benchmark.py`, and link results to TECH_STACK and ARCHITECTURE.

### D. Security and privacy
- Replace `passwords.yml` with `.env` + macOS Keychain integration; document local-only storage and rotation.
- Add a minimal threat model (STRIDE-lite) and data handling policy: PII, dataset residency, local-only operation, and optional redaction tools.
- Document contamination detection: define metric, thresholds, and when to block training.

### E. Reliability, reproducibility, and SLOs
- Define SLOs: training job restart within N minutes after crash; checkpoint every K steps or T minutes; CLI command error rate targets.
- Determinism: define global seeding policy, note sources of nondeterminism, and how to request deterministic runs.
- Artifact lineage: specify output directory structure, run IDs, config checksum, git SHA, and environment capture.

### F. Checkpoint portability spec
- Write a formal “Universal Checkpoint Spec” with: versioned metadata, backend, dtype, param mapping table (PyTorch↔MLX), tokenizer version, and validation checksum.
- Provide migration scripts and compatibility tests.

### G. Architecture diagrams and contracts
- Add C4 Context and Container diagrams for: CLI-only MVP and future CLI+API+UI deployment.
- Provide JSONSchema for `train.yml` and dataset templates; include OpenAPI for planned endpoints; add a CLI argument schema table derived from Typer.

### H. Packaging strategy decision record
- Add an ADR comparing Homebrew vs PyPI vs DMG: target users, update channels, signing/codesign, sandboxing, dependency constraints, and maintenance burden; select a primary channel for Phase 3.

## Document-Specific Feedback and Suggested Edits

### ARCHITECTURE.md
- Status: Replace hard-coded test counts with a link to `STATUS.md` generated by CI.
- Data Pipeline: Limit claimed formats to JSON/JSONL for now; move CSV/TSV/Parquet/HTML to “Planned.”
- UI/API: Retitle “Web Dashboard (Secondary)” and “API Endpoints” as Phase 4 scope; add a note that endpoints are illustrative, not implemented.
- Training Optimization: Add a table clarifying FP16 vs BF16 support in MLX across M-series chips.
- Configuration Strategy: Add JSONSchema link and example `train.yml` with comments.
- Security & Privacy: Add a subsection detailing secrets storage, local-only guarantees, and Keychain support.

### MLX_ARCHITECTURE.md
- Status: Make it match the canonical `STATUS.md` and remove divergent counts.
- Benchmarks: Add a “Methodology” subsection referencing perf harness; include full environment and model details; replace absolute numbers with medians and IQR across N runs.
- Memory Analysis: Replace generic ≈5× rule with optimizer/dtype-specific formulas; distinguish LoRA vs full FT.
- Limitations: Expand operator coverage table with links to fallback modules; add a decision tree for when to switch backends.
- Checkpoint Compatibility: Move example into a formal “Universal Checkpoint Spec” doc and reference it here.

### MLX_INTEGRATION.md
- Integration Steps: Mark completed steps explicitly and link to code locations; move aspirational items to “Planned.”
- Tests: Replace fixed counts with references to test jobs (unit/integration) in CI.
- Real Model Integration: Summarize the MLX module hierarchy workaround; link to the exact functions/classes in code and to unit tests covering it.
- Add “Troubleshooting” section: common errors, how to detect MLX availability, how to validate a conversion.

### PHASE2_PLAN.md
- Convert to a retrospective: what shipped, gaps, and learnings; move future work to Phase 3/4 plans.
- Add acceptance criteria checklists with CI links for each deliverable.
- Include a risk log with items closed vs deferred.

### PROJECT_STRUCTURE.md
- Reflect actual code: remove or mark placeholders for non-existent files (e.g., DB, some API/UI pieces) and clearly tag phase.
- Add ownership per top-level module; include CODEOWNERS mapping.
- Generate sections (Makefile targets, CLI commands) from source-of-truth scripts to avoid drift.

### TECH_STACK.md
- Version pins: specify the exact versions tested in CI and link to lockfiles.
- Quantization: clarify current support vs planned; remove implied PEFT/QLoRA details not implemented yet or mark as planned.
- Monitoring: differentiate required (bundled) vs optional tools; provide a minimal default.

## Prioritized Action Plan (2–4 weeks)

Week 1
- Create `STATUS.md` from CI artifacts; remove hard-coded counts from all docs.
- Add perf methodology doc and basic benchmark harness; update MLX docs to reference it.
- Trim scope claims in ARCHITECTURE/TECH_STACK to match code; mark future items clearly.

Week 2
- Publish JSONSchema for `train.yml` and dataset templates; add deterministic run policy and artifact lineage.
- Draft Universal Checkpoint Spec and add unit tests that validate round-trip MLX↔PyTorch mapping on a small model.

Week 3–4
- Security hardening: replace `passwords.yml` with `.env` + Keychain helper; add threat model and data policy.
- Author ADR on packaging strategy and select a Phase 3 primary channel.
- Add C4 diagrams and API OpenAPI stubs for Phase 4, but keep marked as planned.

## Open Questions

- Do we want CSV/Parquet support in Phase 3, or is JSONL sufficient for target users?
- Is the primary distribution channel PyPI for developers, with Homebrew as a convenience wrapper, and DMG deferred?
- What determinism level is acceptable for training (strict vs best-effort)?
- What are minimal viable evaluation metrics (perplexity only vs task-specific), and where do results live?

## Quick Wins (Low Effort, High Value)

- Add a single “Status & Coverage” badge row to all docs linking to CI dashboards.
- Publish a one-page “Reproducible Benchmark Guide.”
- Add a `--seed` flag to `ft train` docs and show it in examples.
- Include explicit “Supported Today vs Planned” tables in ARCHITECTURE and TECH_STACK.

---

By tightening scope to what exists, formalizing measurement and compatibility, and adding security/reliability guardrails, the design docs will function as an authoritative, durable reference that scales with the project.



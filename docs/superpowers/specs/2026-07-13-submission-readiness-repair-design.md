# Submission Readiness Repair Design

## Goal

Make the MSME Financial Health Card ready for a repeatable hackathon demo while preserving its current FastAPI, LangGraph, RAG, and Next.js architecture.

## Provider Strategy

- Default LLM model: `gemini-3.1-flash-lite`, confirmed available to the configured Gemini account.
- Keep the existing OpenAI-compatible client abstraction. An OpenAI model remains an opt-in configuration using `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
- Do not migrate providers before evidence from the semantic evaluations shows Gemini quality is insufficient.

## Demo Flow

- Make the four fixed personas available in the dashboard for deterministic demonstrations: healthy, new-to-credit, buyer-concentrated, and seasonal.
- Retain the custom-profile form as an additional path, but present it separately from the curated demo path.
- Render the returned grounding trace with the health card so failed narrative checks are visible to the loan officer.

## Backend Integrity

- Correct the documented backend startup command without changing the established `backend.app` import convention.
- Enforce that every accepted `WeightVector` has values in `[0, 1]` and totals `1.0` within a small floating-point tolerance.
- Require complete, non-duplicated weight rationale dimensions. Invalid LLM responses use the existing default weights and rationale.
- Carry `gst_registered` through custom profile generation and reduce the GST evidence signal when it is false.
- Restrict custom sectors to the form's supported set and escape all profile-derived values included in XML-delimited prompts.
- Make retriever construction return an empty result when the persisted index is absent or embedding initialization fails, without downloading/loading an embedding model first.
- Build the local Chroma index from the tracked corpus PDFs for the demo environment.
- Ensure each advertised CFCR stress scenario affects the CFCR calculation for representative profiles, including receivable delay.

## Evaluation Alignment

- Repair the frontend API test around the supported public client methods.
- Update Layer 6 tests to mock the production OpenAI-compatible client call and assert production validation behavior.
- Use `gemini-3.1-flash-lite` as the default DeepEval judge model when the default Gemini endpoint is used.
- Add focused regression tests before each behavior fix, then run backend tests, frontend tests, lint, production build, RAG-backed API smoke checks, and credentialed semantic evaluations where the configured key permits them.

## Non-Goals

- No new database, authentication, real data integrations, or cloud infrastructure.
- No OpenAI migration unless Gemini evaluation evidence identifies a quality issue.
- Do not alter the user-modified `backend/.env.example` outside of an explicit, necessary model-default change.

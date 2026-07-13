# Project: MSME Financial Health Card — Stress-Tested Credit Scoring
**Hackathon:** IDBI Innovate, Track 03 (Financial Inclusion / Digital Lending / Credit Decisioning)
**Core differentiator vs. other teams:** Every other team will build a static aggregator ("combine GST+UPI+AA+EPFO → show one score"). This project stress-tests that score against realistic financial shocks, the same way a bank's LCR stress-tests liquidity — applied to a business borrower instead of the bank itself.

---

## 0. CORE ARCHITECTURE — Risk Management & the LCR Parallel
**This is the headline of the pitch. Everything else in this spec (RAG, weight-setting, personas) is supporting infrastructure that feeds this.**

### The precise analogy (state it this precisely, not looser)
Under Basel III, a bank's **Liquidity Coverage Ratio** = High-Quality Liquid Assets ÷ Total Net Cash Outflows over a 30-day stress period, required to stay ≥ 100%. It answers one question: *can this institution survive a defined liquidity shock without failing?*

This project computes an MSME-equivalent metric — call it the **Cash Flow Coverage Ratio (CFCR)**:
```
CFCR = Liquid Buffer / Projected Net Cash Outflow (stress window)

where:
  Liquid Buffer = avg_account_balance + near-term realizable receivables (from UPI inflow patterns)
  Projected Net Cash Outflow = existing_loan_emi_total + operating_outflow_estimate,
                                re-computed under each stress scenario
  Stress window = 30-60 days (documented choice, analogous to the Basel 30-day window,
                                widened to reflect MSME cash-cycle lumpiness — state this
                                explicitly as a deliberate adaptation, not an oversight)
```
CFCR ≥ 1.0 → the business can absorb the shock without a liquidity failure. CFCR < 1.0 → it cannot, under that scenario, with current buffers.

**Why this framing is defensible in front of bank judges, and analogy alone isn't:** you're not just saying "we do stress-testing like banks do" — you're naming the exact structural components (liquid buffer numerator, stressed outflow denominator, a defined window, a pass/fail threshold) and mapping each one to an MSME-available data field. This is the difference between an analogy a banker can poke holes in and a model a banker can audit.

**What this is NOT:** this is not a literal Basel III LCR calculation, does not use HQLA classifications, and is not regulatory-grade. Say this plainly if asked — overclaiming regulatory equivalence is a bigger credibility risk than acknowledging the adaptation.

### Where this sits architecturally
- CFCR computation lives inside the **Risk Engine (Node 3)** — it is not a separate node, it's the flagship output of the node that already exists in the pipeline.
- The Risk Engine's baseline Financial Health Score and the CFCR are reported together: the health score is the multidimensional "is this business creditworthy" view; CFCR is the narrower "does this business survive a defined shock" view. Both matter; CFCR is the more novel, more defensible, more pitch-worthy number.
- RAG-grounded weight-setting (Section 5, Nodes 1.5a/1.5b) exists to make the Financial Health Score's weighting defensible — it is supporting infrastructure for the score, not the headline. Don't let the pitch or the build spend more time on RAG polish than on making CFCR's stress scenarios and numbers airtight.
- **Build sequencing is unaffected by this framing** — see Section 5 for the actual node build order. This section governs pitch emphasis and what the Risk Engine must compute; it does not reorder what the agent builds first.

---

## 1. Goal
Build a working, demoable full-stack app in a hackathon time-box that:
1. Ingests a synthetic MSME financial profile (GST, UPI, AA-linked bank data, EPFO)
2. Uses an LLM, grounded via RAG over real RBI/MSME lending guidance, to set sector-specific scoring weights **once per profile** (locked before stress-testing — see Section 5, Node 1.5)
3. Computes a baseline multidimensional Financial Health Score using those locked weights
4. Runs that score through 4 stress scenarios and shows how it degrades
5. Explains the results in plain language, citing retrieved guidance where relevant, with every claim (numeric and citation) grounded and verifiable — no hallucinated claims — this is a credit-decisioning context, so ungrounded claims are a real harm vector, not just an accuracy nitpick
6. Presents this as a "Financial Health Card" UI a loan officer could plausibly use

**Time-box priority order if you run short on hours:** synthetic data + deterministic Risk Engine + basic frontend is the non-negotiable core. RAG-grounded weight-setting is the differentiator layer — build it second. If RAG retrieval isn't stable in time for the demo, degrade gracefully to Section 10's fallback rather than risk it breaking live.

## 2. Non-Goals (explicitly out of scope — do not build these)
- No real GST/UPI/AA/EPFO API integration or OAuth flows. All financial profile data is synthetic/mocked.
- No live document scraping. The RAG corpus (Section 4a) is pre-downloaded and pre-embedded once at build time, not fetched at runtime.
- No user authentication system.
- No persistent production database — SQLite or in-memory is sufficient.
- No literal LCR (Liquidity Coverage Ratio) calculation — LCR is a regulatory bank ratio for HQLA vs. net cash outflows over 30 days. We are borrowing the *stress-testing philosophy*, not implementing the formula. Do not let the agent try to compute a literal Basel LCR figure for an MSME — it doesn't map.
- No claim about "buyer concentration risk" beyond what's inferable from mock UPI counterparty clustering. If it can't be grounded in a data field, it doesn't go in the narrative.
- No per-stress-scenario reweighting. Weights are set once per profile (pre-stress) and held constant across all 4 scenarios — this is what keeps the stress test measuring "does this business survive a shock" rather than "did the LLM change its mind mid-test." This was a deliberate call after review — do not let the agent "improve" this by making weights dynamic per scenario.

## 3. Tech Stack
- **Backend:** FastAPI + LangGraph (Python)
- **Frontend:** Next.js 15 (App Router) + Tailwind CSS
- **LLM calls:** Google Gemini 2.5 Pro/Flash via Google AI Studio API, used in Weight-Setter (Node 1.5b), Explainer (Node 4), and the Grounding Validator's LLM fallback (Node 5). Chosen for free-tier cost during the hackathon. **Check Google AI Studio's free-tier rate limits (requests/minute and /day) against your expected total calls per analysis run before demo day** — each full pipeline run makes at least 2-3 LLM calls (weight-setter + explainer + occasional grounding fallback), and rehearsal runs count against the same quota.
- **RAG / retrieval:** ChromaDB (local, embedded, free) + `sentence-transformers` for embeddings — kept local and decoupled from the LLM provider. This is deliberate: embeddings run before every other node in the pipeline, so keeping them free of any API quota or network dependency protects demo stability even though the LLM calls now go to Google's API.
- **Data:** Synthetic generators (Python, numpy/pandas) for financial profiles — no external calls. RAG corpus is real public documents (Section 4a), downloaded once and embedded at build time.
- **Storage:** SQLite (or in-memory dict store if faster to stand up) — persistence is a nice-to-have, not a requirement
- **Deployment target:** local demo only, no cloud infra needed

## 4. Data Model

### MSME Profile (unified schema, output of Data Aggregator)
```
{
  "msme_id": str,
  "business_name": str,
  "sector": str,
  "years_operating": int,
  "gst": {
    "monthly_turnover_series": [float] (12 months),
    "filing_consistency_score": float (0-1),
    "yoy_growth_rate": float
  },
  "upi": {
    "monthly_inflow_series": [float] (12 months),
    "monthly_outflow_series": [float] (12 months),
    "transaction_frequency": int,
    "top_counterparty_share": float (0-1)   // proxy for buyer concentration
  },
  "aa_bank_data": {
    "existing_loan_count": int,
    "existing_loan_emi_total": float,
    "overdraft_utilization_rate": float (0-1),
    "avg_account_balance": float,
    "bounced_payment_count_12mo": int,
    "estimated_monthly_operating_outflow": float   // needed for CFCR denominator, Section 0
  },
  "epfo": {
    "employee_count_series": [int] (12 months),
    "payroll_consistency_score": float (0-1)
  }
}
```

## 4a. RAG Document Corpus
Source 5-10 public documents **before** build starts (this is a manual sourcing task for Tyra, not something to leave to the agent to "find" at runtime):
- RBI circulars/guidelines on MSME lending and credit assessment (search rbi.org.in for MSME-related master circulars)
- RBI/industry guidance on alternate data usage in credit (Account Aggregator framework notes, PSL/MSME classification norms)
- 1-2 sector risk reports if easily available (e.g. manufacturing vs. services vs. agri-adjacent MSME risk profiles)

Process:
1. Download PDFs into `/backend/app/rag/corpus/`
2. Write a one-time preprocessing script (`build_index.py`) that chunks each doc (e.g. ~500 tokens/chunk, some overlap), embeds with `sentence-transformers`, and persists to a local Chroma collection
3. This index is built once and committed/cached — do not re-embed on every server start
4. Each stored chunk must retain its source document name and section, so citations in the narrative can point back to something real, not just "RBI guidance says..." with no traceable source

### Persona set (build at least these 4 for demo variety)
1. **Healthy & bankable** — stable growth, low concentration, clean repayment history
2. **New-to-Credit (NTC) thin-file** — strong UPI/GST signal but no existing loan history to anchor a traditional score
3. **Buyer-concentrated risk** — good numbers overall, but >60% revenue from one UPI counterparty
4. **Seasonal/volatile** — legitimate business (e.g. agri-adjacent) with high month-to-month variance that a naive average would misread as unstable

## 5. LangGraph Pipeline

```
[Data Aggregator] → [Sector Context Retriever (RAG)] → [Weight-Setter (LLM)] → [Stress Scenario Generator] → [Risk Engine (tool call)] → [Explainer (LLM + RAG)] → [Grounding Validator] → Output
```

### Node 1: Data Aggregator
- Input: msme_id (or raw synthetic generation params)
- Output: unified MSME Profile (schema above)
- Pulls from the 4 mock sources and merges into one object. Pure data-shaping, no LLM call needed here.

### Node 1.5a: Sector Context Retriever (RAG)
- Input: MSME Profile (specifically sector, years_operating, thin-file/NTC flag)
- Output: top-k retrieved chunks from the Chroma index (Section 4a) relevant to this business's sector/profile type
- Pure retrieval, no LLM call. Query the vector store with a query built from profile metadata (e.g. "MSME credit risk assessment [sector] alternate data").

### Node 1.5b: Weight-Setter (LLM, RAG-grounded)
- Input: MSME Profile + retrieved chunks from Node 1.5a
- Output: a single locked set of scoring weights (GST, UPI, AA, EPFO dimensions) for this profile, plus a short rationale citing which retrieved chunk(s) informed each weight decision
- **This runs once per profile, before any stress scenario is applied. Weights are then locked and reused unchanged across all 4 stress scenarios** — see Section 2 Non-Goals for why this is non-negotiable.
- Prompt constraint: the model must justify each weight only by reference to the MSME Profile fields and the retrieved chunks — no unsourced reasoning. If no retrieved chunk supports a particular weighting choice, the model should fall back to a documented default and say so explicitly, not invent a justification.
- Output feeds directly into Risk Engine (Node 3) as the weight vector, and the rationale + citations feed the Grounding Validator (Node 6).

### Node 2: Stress Scenario Generator
- Input: MSME Profile
- Output: list of scenario objects, each a structured perturbation:
  1. `receivable_delay_60d` — simulate a large receivable pushed 60 days out (reduce near-term UPI inflow, spike overdraft utilization)
  2. `revenue_drop_20pct` — apply a flat 20% cut to GST turnover + UPI inflow series
  3. `buyer_loss` — zero out the top counterparty's UPI inflow share (only meaningful if `top_counterparty_share` is non-trivial)
  4. `rate_hike` — increase `existing_loan_emi_total` by a configurable % (simulate a floating-rate loan repricing)
- This node can be deterministic (no LLM needed) — it's just parameterized data transforms. Don't over-engineer with an LLM call here; save the LLM budget for Explainer.

### Node 3: Risk Engine (tool call, deterministic code — not an LLM prompt)
- Input: MSME Profile + locked weight vector (from Node 1.5b) + list of stress scenarios
- Computes:
  - **Cash Flow Coverage Ratio (CFCR)** — the flagship metric, per Section 0. Baseline CFCR + a recomputed CFCR under each of the 4 stress scenarios, using the formula defined in Section 0. This is the number the pitch leads with.
  - **Baseline Financial Health Score** (0-100): weighted composite across GST trend, UPI cash-flow stability, AA repayment behavior, EPFO payroll consistency, using the weights locked by Node 1.5b. Log which weights were used alongside the score.
  - **Cash-flow volatility**: coefficient of variation on UPI monthly net cash flow series
  - **Buyer concentration risk flag**: derived directly from `top_counterparty_share`, thresholded (e.g. >40% = flag)
  - **Stressed score per scenario**: re-run the same scoring formula (same locked weights, perturbed inputs) for each of the 4 scenarios, alongside the stressed CFCR
- Output: structured JSON with CFCR (baseline + per-scenario), health score, weights used, per-scenario stressed scores, and the specific numeric deltas that drove each change. This structured output is what the Grounding Validator will check the Explainer against.

### Node 4: Explainer (LLM call + RAG)
- Input: Risk Engine's structured output + a fresh retrieval pass against the RAG corpus for narrative-relevant context (e.g. citing the specific guidance that justifies treating a stress scenario as material risk)
- Output: Financial Health Card narrative — strengths, risks, "what breaks under which scenario," in loan-officer-readable language, with inline citations to retrieved guidance where relevant
- Prompt constraints:
  - Every quantitative claim must trace to a field in the Risk Engine output — no numbers not present in the input JSON
  - Every regulatory/guidance-style claim must trace to an actually-retrieved chunk — no citing "RBI guidelines" or similar without a specific retrieved source backing it
  - If nothing relevant was retrieved for a given point, the narrative should state the point without a citation rather than inventing one

### Node 5 (final): Grounding Validator (deterministic, code — LLM only as fallback)
- Input: Explainer's narrative + Risk Engine's structured output + the retrieved RAG chunks actually used in Node 4
- Logic: two checks, both deterministic —
  1. **Numeric grounding**: extract numeric claims from the narrative (regex/simple parsing is fine for a hackathon), check each against the source JSON within a tolerance
  2. **Citation grounding**: for each cited claim, confirm it maps to a chunk that was actually retrieved (not a document name invented by the model)
  - If everything passes, pass through. If either check fails, only then invoke a cheap LLM call to flag/rewrite the offending sentence.
- Rationale (this is your strongest pitch point): treating this as a validation layer instead of a reasoning node keeps latency low and avoids a new LLM-failure surface right before a live demo, while giving judges a concrete, two-part answer to "how do you prevent a hallucinated number *or* a hallucinated citation from feeding a credit decision."
- Output: final narrative + a "grounding trace" object covering both numeric and citation checks (which claims were checked, pass/fail, and source) — this trace should be visible in the UI, it's your strongest talking point.

## 6. Backend API Contract (FastAPI)
- `POST /api/personas` — returns the list of pre-built demo personas
- `POST /api/msme/{persona_id}/analyze` — runs the full LangGraph pipeline for a given persona, returns:
  ```
  {
    "profile_summary": {...},
    "cfcr_baseline": float,
    "cfcr_by_scenario": [{scenario, cfcr, pass_fail}],
    "weights_used": {gst: float, upi: float, aa: float, epfo: float},
    "weight_rationale": [{dimension, reasoning, cited_chunk_id}],
    "baseline_score": float,
    "stress_results": [{scenario, stressed_score, delta, key_drivers}],
    "narrative": str,
    "grounding_trace": [{claim, type: "numeric"|"citation", source_field_or_chunk, status}]
  }
  ```
- `GET /api/msme/{persona_id}/report` — cached retrieval of last analysis (optional, nice-to-have)

## 7. Frontend Requirements (Next.js)
- **Persona selector** — pick from the 4 demo personas
- **CFCR headline display** — the largest, most prominent number on the page. Show baseline CFCR with a clear pass/fail (≥1.0 vs <1.0) indicator, framed explicitly as "Cash Flow Coverage Ratio — the MSME equivalent of a bank's LCR." This is the first thing a judge should see.
- **Financial Health Card view** — baseline health score, strengths/risks summary (secondary to CFCR, not competing for the same visual weight)
- **Stress test panel** — 4 scenario toggles/sliders, each showing the CFCR *and* health score delta live (bar chart or gauge, recharts is fine) — CFCR crossing below 1.0 under a scenario should be visually unmistakable (e.g. red threshold line)
- **Grounding trace panel** — visible list of narrative claims with pass/fail against source data, tagged as numeric or citation
- **Weight rationale panel** — supporting detail, can be a secondary/expandable panel rather than headline real estate — show the locked weights and cited reasoning, but don't let it compete visually with CFCR
- Keep styling clean and credible — this is pitching to a bank, not a consumer app. Avoid generic AI-startup gradient aesthetics.

## 8. File Structure (suggested)
```
/backend
  /app
    /graph
      nodes.py          # aggregator, context_retriever, weight_setter, stress_generator, explainer, grounding_validator
      risk_engine.py    # deterministic scoring logic (tool)
      pipeline.py        # LangGraph graph definition
    /rag
      corpus/             # downloaded source PDFs (Section 4a)
      build_index.py      # one-time chunk + embed + persist to Chroma
      retriever.py         # query interface used by Node 1.5a and Node 4
    /data
      mock_generators.py
      personas.py         # 4 predefined persona profiles
    /schemas
      models.py           # pydantic models for MSME Profile, Risk Output, etc.
    main.py                # FastAPI app + routes
  requirements.txt
/frontend
  /app
    page.tsx
    /components
      PersonaSelector.tsx
      HealthCard.tsx
      StressPanel.tsx
      GroundingTrace.tsx
  package.json
MASTER_SPEC.md (this file)
```

## 9. Demo Script (build toward this)
1. Open on "Buyer-concentrated risk" persona — lead with the CFCR headline: "this is the equivalent of a bank's LCR, applied to the business" — baseline CFCR is comfortably above 1.0
2. Toggle "buyer_loss" stress scenario — CFCR visibly drops, potentially below the 1.0 threshold — this is the moment: "under Basel III, a bank in this position would fail its liquidity requirement; this business has the same problem, and no one currently measures it this way"
3. Only then show the Financial Health Card and weight rationale as supporting detail — "and every input to this is grounded and traceable" — this is where RAG/grounding earns its place, after CFCR has already made the case
4. Toggle a second scenario to show compounding stress
5. Point at the grounding trace panel: "this checks two things — does every number trace back to real data, and does every citation trace back to a document we actually retrieved, not one the model made up"


## 10. Open Decisions for Tyra (flagged, not resolved by the agent)
- Final list of 5-10 RAG source documents — needs to be sourced and downloaded before the agent starts building Node 1.5a/1.5b
- Whether to add a 5th persona live during Q&A to show generalizability
- Naming for the product (spec uses "Financial Health Card" as the working name — pitch may want something sharper)
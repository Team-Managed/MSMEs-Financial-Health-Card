"use client";

import { startTransition, useState } from "react";
import dynamic from "next/dynamic";
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  ChartNoAxesCombined,
  CircleHelp,
  Database,
  FileCheck2,
  GitBranch,
  Landmark,
  LockKeyhole,
  PencilLine,
  Scale,
  SearchCheck,
  ShieldCheck,
} from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";
import { analyzeCustom } from "@/lib/api";
import ProfileForm, {
  ARTIFACT_LABELS,
  type AnalysisParams,
  type ArtifactKey,
} from "./ProfileForm";
import HealthCard from "./HealthCard";
import WeightRationale from "./WeightRationale";
import GroundingTrace from "./GroundingTrace";

const StressPanel = dynamic(() => import("./StressPanel"), { ssr: false });
const PIPELINE_STEPS = [
  "Profile signals",
  "Sector guidance",
  "Stress engine",
  "Grounding check",
];

const FEATURES = [
  {
    icon: ChartNoAxesCombined,
    number: "01",
    title: "Cash-flow resilience",
    copy: "CFCR measures whether recurring inflows can cover obligations before and after defined shocks.",
  },
  {
    icon: GitBranch,
    number: "02",
    title: "Scenario stress testing",
    copy: "Receivable delays, revenue contraction, buyer loss, and rate pressure are applied consistently.",
  },
  {
    icon: Database,
    number: "03",
    title: "Sector-grounded context",
    copy: "Retrieved guidance informs source weighting while unsupported recommendations fall back safely.",
  },
  {
    icon: SearchCheck,
    number: "04",
    title: "Claim-level audit",
    copy: "Every reported number resolves to the risk engine and every citation resolves to retrieved evidence.",
  },
];

const PROCESS_STEPS = [
  [
    "Capture",
    "Enter the enterprise profile and operating signals available at assessment time.",
  ],
  [
    "Compute",
    "The deterministic engine calculates baseline health and four comparable stress outcomes.",
  ],
  [
    "Ground",
    "Sector evidence is retrieved and generated claims are checked against source fields.",
  ],
  [
    "Review",
    "The loan officer receives a health card, rationale, stress matrix, and evidence ledger.",
  ],
];

const FAQS = [
  [
    "Does this approve or reject a loan?",
    "No. The Financial Health Card is decision support for an authorised loan officer. It does not replace bank policy, bureau checks, KYC, due diligence, or delegated credit authority.",
  ],
  [
    "What is CFCR?",
    "Cash Flow Coverage Ratio compares recurring operating inflows with expected obligations. A value above 1.0 indicates modeled coverage, while values below 1.0 signal a liquidity shortfall under that scenario.",
  ],
  [
    "Why can an evidence check fail?",
    "A check fails when a generated number does not match its risk-engine field or a citation does not resolve to a retrieved source. Failed claims are excluded from the final narrative.",
  ],
  [
    "How are sector documents used?",
    "Relevant passages are retrieved to inform weighting rationale and explanation. Retrieval does not change deterministic formulas, and unsupported model output is rejected.",
  ],
  [
    "Is the result a credit score?",
    "It is a financial-health indicator built for this assessment workflow. It is not a bureau score and should be interpreted with the underlying cash-flow, stress, and evidence views.",
  ],
];

function AppHeader({
  onReset,
  hasResult,
}: {
  onReset: () => void;
  hasResult: boolean;
}) {
  return (
    <header className="app-header">
      <button
        className="brand"
        type="button"
        onClick={onReset}
        aria-label="Return to assessment start"
      >
        <span className="brand-mark">
          <Landmark size={19} />
        </span>
        <span>
          <strong>Financial Health Card</strong>
          <small>Indicative MSME screening</small>
        </span>
      </button>
      <div className="header-meta">
        {!hasResult ? (
          <nav className="site-nav" aria-label="Landing page navigation">
            <a href="#features">Features</a>
            <a href="#how-it-works">How it works</a>
            <a href="#policy">Policy</a>
            <a href="#faqs">FAQs</a>
          </nav>
        ) : null}
        <span className="system-status">
          <i /> Risk engine online
        </span>
      </div>
    </header>
  );
}

function LandingDetails() {
  return (
    <div className="landing-details">
      <section className="landing-band feature-band" id="features">
        <div className="band-heading">
          <div>
            <p className="eyebrow">Underwriting toolkit</p>
            <h2>One assessment. Four defensible views.</h2>
          </div>
          <p>
            Designed for loan officers who need to understand the reason behind
            a result, not just receive another opaque score.
          </p>
        </div>
        <div className="feature-grid">
          {FEATURES.map(({ icon: Icon, number, title, copy }) => (
            <article key={title}>
              <div className="feature-index">
                <Icon size={20} />
                <span>{number}</span>
              </div>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-band process-band" id="how-it-works">
        <div className="band-heading">
          <div>
            <p className="eyebrow">How it works</p>
            <h2>From operating signals to reviewable evidence.</h2>
          </div>
          <a className="text-link" href="#assessment">
            Start an assessment <ArrowRight size={15} />
          </a>
        </div>
        <ol className="process-list">
          {PROCESS_STEPS.map(([title, copy], index) => (
            <li key={title}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <div>
                <h3>{title}</h3>
                <p>{copy}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="policy-band" id="policy">
        <div className="policy-intro">
          <p className="eyebrow">Decision policy</p>
          <h2>Human judgment remains the final control.</h2>
          <p>
            The Health Card supports assessment; it does not automate sanction,
            pricing, eligibility, or adverse-action decisions.
          </p>
        </div>
        <div className="policy-points">
          <article>
            <Scale size={20} />
            <h3>Decision support only</h3>
            <p>
              Use alongside applicable credit policy, KYC, bureau data, and
              delegated authority.
            </p>
          </article>
          <article>
            <LockKeyhole size={20} />
            <h3>Minimum necessary data</h3>
            <p>
              Only assessment inputs required by the model should be entered;
              avoid personal or secret data.
            </p>
          </article>
          <article>
            <BadgeCheck size={20} />
            <h3>Fail-closed evidence</h3>
            <p>
              Unsupported generated claims are removed and deterministic
              risk-engine facts take precedence.
            </p>
          </article>
        </div>
      </section>

      <section className="landing-band faq-band" id="faqs">
        <div className="faq-heading">
          <CircleHelp size={22} />
          <p className="eyebrow">Frequently asked questions</p>
          <h2>Before you rely on the card.</h2>
        </div>
        <div className="faq-list">
          {FAQS.map(([question, answer]) => (
            <details key={question}>
              <summary>{question}</summary>
              <p>{answer}</p>
            </details>
          ))}
        </div>
      </section>

      <footer className="landing-footer">
        <div className="footer-brand">
          <span className="brand-mark">
            <Landmark size={18} />
          </span>
          <span>
            <strong>Financial Health Card</strong>
            <small>MSME credit intelligence</small>
          </span>
        </div>
        <p>Built for transparent, evidence-led credit review.</p>
        <a href="#assessment">Return to assessment</a>
      </footer>
    </div>
  );
}

export default function Dashboard() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [submittedArtifacts, setSubmittedArtifacts] = useState<Record<
    ArtifactKey,
    boolean
  > | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const worstScenario = result?.cfcr_by_scenario
    .filter((item) => item.scenario !== "baseline")
    .reduce(
      (current, item) => (item.cfcr < current.cfcr ? item : current),
      result.cfcr_by_scenario.find((item) => item.scenario !== "baseline")!,
    );

  async function handleSubmit(params: AnalysisParams) {
    setError(null);
    setLoading(true);
    try {
      const data = await analyzeCustom(params);
      startTransition(() => {
        setSubmittedArtifacts(params.verifiedArtifacts);
        setResult(data);
      });
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "The assessment could not be completed.",
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setResult(null);
    setSubmittedArtifacts(null);
    setError(null);
  }

  const artifactEntries = Object.entries(ARTIFACT_LABELS) as [
    ArtifactKey,
    string,
  ][];
  const verifiedArtifacts = artifactEntries.filter(
    ([key]) => submittedArtifacts?.[key],
  );
  const missingArtifacts = artifactEntries.filter(
    ([key]) => !submittedArtifacts?.[key],
  );

  return (
    <main className="app-shell">
      <AppHeader onReset={reset} hasResult={result !== null} />
      {error ? (
        <div className="error-banner" role="alert">
          <strong>Analysis unavailable.</strong> {error}
        </div>
      ) : null}

      {result === null && !loading ? (
        <>
          <section className="landing-view" id="assessment">
            <div className="landing-copy">
              <p className="eyebrow">IDBI Innovate · Credit decisioning</p>
              <h1>
                MSME Financial
                <br />
                Health Card
              </h1>
              <p className="landing-lede">
                A cash-flow-first underwriting workspace that tests whether a
                business can absorb defined shocks before credit is extended.
              </p>
              <div className="method-strip" aria-label="Assessment method">
                <div>
                  <Activity size={18} />
                  <span>
                    <strong>4 scenarios</strong>
                    <small>Deterministic stress tests</small>
                  </span>
                </div>
                <div>
                  <Database size={18} />
                  <span>
                    <strong>Grounded weights</strong>
                    <small>Sector guidance via RAG</small>
                  </span>
                </div>
                <div>
                  <FileCheck2 size={18} />
                  <span>
                    <strong>Auditable output</strong>
                    <small>Claim-level verification</small>
                  </span>
                </div>
              </div>
              <div className="signal-preview" aria-hidden="true">
                <div className="preview-label">
                  <span>Illustrative liquidity resilience</span>
                  <strong>CFCR 1.42</strong>
                </div>
                <div className="preview-chart">
                  {[72, 82, 76, 91, 86, 64, 58, 44, 67, 73, 52, 61].map(
                    (height, index) => (
                      <i key={index} style={{ height: `${height}%` }} />
                    ),
                  )}
                  <span className="threshold-line">1.0 threshold</span>
                </div>
              </div>
            </div>
            <div className="intake-panel">
              <ProfileForm onSubmit={handleSubmit} loading={loading} />
            </div>
          </section>
          <LandingDetails />
        </>
      ) : null}

      {loading ? (
        <section className="loading-view" aria-live="polite">
          <span className="loading-seal">
            <ShieldCheck size={28} />
          </span>
          <p className="eyebrow">Assessment in progress</p>
          <h2>Testing financial resilience</h2>
          <div className="pipeline-progress">
            {PIPELINE_STEPS.map((step, index) => (
              <span key={step}>
                <i style={{ animationDelay: `${index * 450}ms` }} />
                {step}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      {result !== null && !loading ? (
        <section className="workspace">
          <div className="workspace-title">
            <div>
              <p className="eyebrow">Indicative screening workspace</p>
              <h1>
                {String(
                  result.profile_summary.business_name ?? "MSME assessment",
                )}
              </h1>
            </div>
            <div className="workspace-actions">
              <div className="decision-stamp">
                <ShieldCheck size={16} /> Engine validated
              </div>
              <button className="quiet-action" type="button" onClick={reset}>
                <PencilLine size={15} /> Edit inputs
              </button>
            </div>
          </div>
          <div className="borrower-context" aria-label="Borrower context">
            <div>
              <span>Sector</span>
              <strong>
                {String(result.profile_summary.sector ?? "Not provided")}
              </strong>
            </div>
            <div>
              <span>Operating history</span>
              <strong>
                {String(result.profile_summary.years_operating ?? "–")} years
              </strong>
            </div>
            <div>
              <span>Assessment ID</span>
              <strong>
                {String(result.profile_summary.msme_id ?? "Generated profile")}
              </strong>
            </div>
          </div>
          <aside
            className="approval-readiness"
            aria-label="Approval requirements"
          >
            <div className="readiness-heading">
              <ShieldCheck size={18} />
              <span>
                <strong>
                  {missingArtifacts.length === 0
                    ? "Documents marked available"
                    : `${missingArtifacts.length} verified artifact${missingArtifacts.length === 1 ? "" : "s"} still required`}
                </strong>
                <small>
                  Applicant declarations are not bank verification or approval.
                </small>
              </span>
            </div>
            <div className="artifact-status-groups">
              {verifiedArtifacts.length > 0 ? (
                <section>
                  <strong>Marked available</strong>
                  <ul>
                    {verifiedArtifacts.map(([key, label]) => (
                      <li key={key}>
                        <BadgeCheck size={14} />
                        {label}
                      </li>
                    ))}
                  </ul>
                </section>
              ) : null}
              {missingArtifacts.length > 0 ? (
                <section>
                  <strong>Still required for verification</strong>
                  <ul>
                    {missingArtifacts.map(([key, label]) => (
                      <li key={key}>
                        <CircleHelp size={14} />
                        {label}
                      </li>
                    ))}
                  </ul>
                </section>
              ) : null}
            </div>
          </aside>
          <div className="workspace-main">
            <HealthCard data={result} />
            <StressPanel data={result} />
            <article className="analysis-panel decision-brief">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Loan officer brief</p>
                  <h2>What the assessment indicates</h2>
                </div>
                <span>Source-validated</span>
              </div>
              <div className="brief-copy">
                <p>
                  Baseline cash flow coverage is{" "}
                  <strong>{result.cfcr_baseline.toFixed(2)}x</strong>, which{" "}
                  {result.cfcr_baseline >= 1 ? "clears" : "falls below"} the
                  1.00 minimum.{" "}
                  {worstScenario
                    ? `The most severe modeled condition is ${worstScenario.scenario.replaceAll("_", " ")}, where coverage moves to ${worstScenario.cfcr.toFixed(2)}x. `
                    : ""}
                  The composite financial health score is{" "}
                  <strong>{result.baseline_score.toFixed(0)}/100</strong>.
                </p>
                {result.narrative ? (
                  <details className="raw-commentary">
                    <summary>View validated generated commentary</summary>
                    <p>{result.narrative}</p>
                  </details>
                ) : null}
              </div>
            </article>
            <section className="methodology-section">
              <div className="methodology-heading">
                <div>
                  <p className="eyebrow">Methodology and controls</p>
                  <h2>Review the machinery behind the result</h2>
                </div>
                <p>
                  Open these only when you need weighting rationale or
                  claim-level evidence.
                </p>
              </div>
              <div className="methodology-grid">
                <WeightRationale data={result} />
                <GroundingTrace data={result} />
              </div>
            </section>
          </div>
        </section>
      ) : null}
    </main>
  );
}

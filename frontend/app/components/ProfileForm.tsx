"use client";

import { useReducer, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Building2,
  FileCheck2,
  ShieldCheck,
} from "lucide-react";

const SECTORS = [
  ["manufacturing", "Manufacturing"],
  ["services", "Services"],
  ["textiles", "Textiles"],
  ["agri-processing", "Agri-processing"],
  ["trading", "Trading"],
  ["food-processing", "Food processing"],
] as const;

type Tier = "micro" | "small" | "medium";
type Binary = "yes" | "no";
export type ArtifactKey =
  | "bankStatements"
  | "gstItr"
  | "promoterKyc"
  | "businessIdentity"
  | "financialStatements";

export const ARTIFACT_LABELS: Record<ArtifactKey, string> = {
  bankStatements: "12-month business bank statement matched to GST returns",
  gstItr: "Filed GST returns and latest ITR with consent",
  promoterKyc: "PAN and Aadhaar of promoters",
  businessIdentity: "Entity PAN, sister-concern PAN, and Udyam registration",
  financialStatements:
    "Audited financials and current-year provisional statements",
};

interface FormState {
  sector: string;
  years: string;
  turnoverRange: Tier | "";
  gstRegistered: Binary | "";
  employeeTier: Tier | "";
  requestedAmount: string;
  interestRate: string;
  utilization: string;
  annualTurnover: string;
  monthlyInflow: string;
  monthlyOutflow: string;
  bankBalance: string;
  existingEmi: string;
  topBuyerShare: string;
  bouncedPayments: string;
  filingConsistency: string;
  yoyGrowth: string;
  seasonal: Binary | "";
}

type Action = { field: keyof FormState; value: string };

const INITIAL: FormState = {
  sector: "",
  years: "",
  turnoverRange: "",
  gstRegistered: "",
  employeeTier: "",
  requestedAmount: "",
  interestRate: "12",
  utilization: "75",
  annualTurnover: "",
  monthlyInflow: "",
  monthlyOutflow: "",
  bankBalance: "",
  existingEmi: "0",
  topBuyerShare: "",
  bouncedPayments: "0",
  filingConsistency: "",
  yoyGrowth: "",
  seasonal: "",
};

export interface AnalysisParams {
  sector: string;
  yearsOperating: number;
  profileType: string;
  msmeTier: Tier;
  gstRegistered: boolean;
  employeeTier: Tier;
  requestedAmountLakh: number;
  annualInterestRatePct: number;
  expectedUtilizationPct: number;
  annualTurnoverLakh: number;
  avgMonthlyInflowLakh: number;
  avgMonthlyOperatingOutflowLakh: number;
  avgBankBalanceLakh: number;
  existingMonthlyEmiLakh: number;
  topBuyerSharePct: number;
  bouncedPayments12mo: number;
  gstFilingConsistencyPct: number;
  yoyGrowthPct: number;
  verifiedArtifacts: Record<ArtifactKey, boolean>;
}

interface Props {
  onSubmit: (params: AnalysisParams) => void;
  loading: boolean;
  compact?: boolean;
}

function reducer(state: FormState, action: Action): FormState {
  return { ...state, [action.field]: action.value };
}

function profileType(form: FormState) {
  if (Number(form.existingEmi) === 0) return "ntc";
  if (Number(form.topBuyerShare) >= 50) return "buyer_concentrated";
  if (form.seasonal === "yes") return "seasonal";
  return "healthy";
}

function stepComplete(form: FormState, step: number) {
  if (step === 1) {
    return (
      Boolean(
        form.sector && form.years && form.turnoverRange && form.employeeTier,
      ) && form.gstRegistered === "yes"
    );
  }
  if (step === 2) {
    return [
      "requestedAmount",
      "interestRate",
      "utilization",
      "annualTurnover",
      "monthlyInflow",
      "monthlyOutflow",
      "bankBalance",
    ].every((field) => Number(form[field as keyof FormState]) > 0);
  }
  return (
    [
      "existingEmi",
      "topBuyerShare",
      "bouncedPayments",
      "filingConsistency",
      "yoyGrowth",
    ].every((field) => form[field as keyof FormState] !== "") &&
    Boolean(form.seasonal)
  );
}

function SegmentedControl<T extends string>({
  name,
  value,
  options,
  onChange,
}: {
  name: string;
  value: T | "";
  options: readonly { value: T; label: string; detail?: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented-control" role="radiogroup" aria-label={name}>
      {options.map((option) => (
        <label
          key={option.value}
          className={value === option.value ? "is-selected" : ""}
        >
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span>{option.label}</span>
          {option.detail ? <small>{option.detail}</small> : null}
        </label>
      ))}
    </div>
  );
}

function NumberField({
  label,
  suffix,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  suffix?: string;
  value: string;
  min: string;
  max?: string;
  step?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field numeric-field">
      <span>{label}</span>
      <span className="input-with-suffix">
        <input
          type="number"
          min={min}
          max={max}
          step={step ?? "0.01"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
        {suffix ? <small>{suffix}</small> : null}
      </span>
    </label>
  );
}

export default function ProfileForm({
  onSubmit,
  loading,
  compact = false,
}: Props) {
  const [form, dispatch] = useReducer(reducer, INITIAL);
  const [step, setStep] = useState(1);
  const [artifacts, setArtifacts] = useState<Record<ArtifactKey, boolean>>({
    bankStatements: false,
    gstItr: false,
    promoterKyc: false,
    businessIdentity: false,
    financialStatements: false,
  });
  const complete =
    stepComplete(form, 1) && stepComplete(form, 2) && stepComplete(form, 3);
  const set = (field: keyof FormState, value: string) =>
    dispatch({ field, value });

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!complete || loading) return;
    onSubmit({
      sector: form.sector,
      yearsOperating: Number(form.years),
      profileType: profileType(form),
      msmeTier: form.turnoverRange as Tier,
      gstRegistered: form.gstRegistered === "yes",
      employeeTier: form.employeeTier as Tier,
      requestedAmountLakh: Number(form.requestedAmount),
      annualInterestRatePct: Number(form.interestRate),
      expectedUtilizationPct: Number(form.utilization),
      annualTurnoverLakh: Number(form.annualTurnover),
      avgMonthlyInflowLakh: Number(form.monthlyInflow),
      avgMonthlyOperatingOutflowLakh: Number(form.monthlyOutflow),
      avgBankBalanceLakh: Number(form.bankBalance),
      existingMonthlyEmiLakh: Number(form.existingEmi),
      topBuyerSharePct: Number(form.topBuyerShare),
      bouncedPayments12mo: Number(form.bouncedPayments),
      gstFilingConsistencyPct: Number(form.filingConsistency),
      yoyGrowthPct: Number(form.yoyGrowth),
      verifiedArtifacts: artifacts,
    });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className={compact ? "profile-form is-compact" : "profile-form"}
    >
      <div className="form-heading">
        <span className="form-icon">
          <Building2 size={18} />
        </span>
        <div>
          <p className="eyebrow">Indicative pre-screening</p>
          <h2>Business profile</h2>
        </div>
      </div>
      {!compact ? (
        <div className="screening-notice">
          <strong>Not a loan application</strong>
          <span>
            These inputs select a synthetic risk profile. No approval,
            eligibility, limit, or pricing decision is produced.
          </span>
        </div>
      ) : null}

      <div className="intake-steps" aria-label="Application progress">
        {["Business", "Facility & cash flow", "Risk & documents"].map(
          (label, index) => (
            <button
              key={label}
              type="button"
              className={
                step === index + 1
                  ? "is-current"
                  : step > index + 1
                    ? "is-complete"
                    : ""
              }
              onClick={() => index + 1 < step && setStep(index + 1)}
            >
              <span>{index + 1}</span>
              {label}
            </button>
          ),
        )}
      </div>

      {step === 1 ? (
        <fieldset>
          <legend>Business details</legend>
          <div className="form-grid">
            <label className="field">
              <span>Sector</span>
              <select
                value={form.sector}
                onChange={(event) => set("sector", event.target.value)}
              >
                <option value="">Select sector</option>
                {SECTORS.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Years operating</span>
              <input
                type="number"
                min="0"
                max="99"
                placeholder="5"
                value={form.years}
                onChange={(event) => set("years", event.target.value)}
              />
            </label>
          </div>
          <div className="field-group">
            <span>MSME classification</span>
            <SegmentedControl
              name="turnover range"
              value={form.turnoverRange}
              options={[
                { value: "micro", label: "Micro", detail: "Below ₹5 Cr" },
                { value: "small", label: "Small", detail: "₹5–50 Cr" },
                { value: "medium", label: "Medium", detail: "₹50–250 Cr" },
              ]}
              onChange={(value) => set("turnoverRange", value)}
            />
          </div>
        </fieldset>
      ) : null}

      {step === 1 ? (
        <fieldset>
          <legend>Operating signals</legend>
          <div className="form-grid">
            <div className="field-group">
              <span>GST registered</span>
              <SegmentedControl
                name="GST registered"
                value={form.gstRegistered}
                options={[
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" },
                ]}
                onChange={(value) => set("gstRegistered", value)}
              />
            </div>
            <div className="field-group">
              <span>Employees</span>
              <SegmentedControl
                name="employee tier"
                value={form.employeeTier}
                options={[
                  { value: "micro", label: "1–10" },
                  { value: "small", label: "11–50" },
                  { value: "medium", label: "51–250" },
                ]}
                onChange={(value) => set("employeeTier", value)}
              />
            </div>
          </div>
        </fieldset>
      ) : null}

      {step === 2 ? (
        <>
          <fieldset>
            <legend>IDBI i-MSME Express facility</legend>
            <div className="form-grid form-grid-three">
              <NumberField
                label="Cash credit requested"
                suffix="₹ lakh"
                value={form.requestedAmount}
                min="1"
                max="25"
                onChange={(value) => set("requestedAmount", value)}
              />
              <NumberField
                label="Indicative annual rate"
                suffix="%"
                value={form.interestRate}
                min="1"
                max="40"
                step="0.1"
                onChange={(value) => set("interestRate", value)}
              />
              <NumberField
                label="Expected utilisation"
                suffix="%"
                value={form.utilization}
                min="0"
                max="100"
                onChange={(value) => set("utilization", value)}
              />
            </div>
            <p className="field-note">
              Official product range: ₹1–25 lakh cash credit, 12-month tenure.
              Rate remains indicative.
            </p>
          </fieldset>
          <fieldset>
            <legend>Measured financials</legend>
            <div className="form-grid">
              <NumberField
                label="Annual turnover"
                suffix="₹ lakh"
                value={form.annualTurnover}
                min="0.01"
                step="0.01"
                onChange={(value) => set("annualTurnover", value)}
              />
              <NumberField
                label="Average monthly inflow"
                suffix="₹ lakh"
                value={form.monthlyInflow}
                min="0.01"
                step="0.01"
                onChange={(value) => set("monthlyInflow", value)}
              />
              <NumberField
                label="Monthly operating outflow"
                suffix="₹ lakh"
                value={form.monthlyOutflow}
                min="0"
                step="0.01"
                onChange={(value) => set("monthlyOutflow", value)}
              />
              <NumberField
                label="Average bank balance"
                suffix="₹ lakh"
                value={form.bankBalance}
                min="0"
                step="0.01"
                onChange={(value) => set("bankBalance", value)}
              />
            </div>
          </fieldset>
        </>
      ) : null}

      {step === 3 ? (
        <>
          <fieldset>
            <legend>Repayment and operating risk</legend>
            <div className="form-grid">
              <NumberField
                label="Existing monthly EMI"
                suffix="₹ lakh"
                value={form.existingEmi}
                min="0"
                step="0.01"
                onChange={(value) => set("existingEmi", value)}
              />
              <NumberField
                label="Top buyer revenue share"
                suffix="%"
                value={form.topBuyerShare}
                min="0"
                max="100"
                onChange={(value) => set("topBuyerShare", value)}
              />
              <NumberField
                label="Payment bounces, last 12m"
                value={form.bouncedPayments}
                min="0"
                step="1"
                onChange={(value) => set("bouncedPayments", value)}
              />
              <NumberField
                label="GST filing consistency"
                suffix="%"
                value={form.filingConsistency}
                min="0"
                max="100"
                onChange={(value) => set("filingConsistency", value)}
              />
              <NumberField
                label="Year-on-year growth"
                suffix="%"
                value={form.yoyGrowth}
                min="-100"
                max="500"
                step="0.1"
                onChange={(value) => set("yoyGrowth", value)}
              />
              <div className="field-group">
                <span>Material seasonality</span>
                <SegmentedControl
                  name="seasonal"
                  value={form.seasonal}
                  options={[
                    { value: "yes", label: "Yes" },
                    { value: "no", label: "No" },
                  ]}
                  onChange={(value) => set("seasonal", value)}
                />
              </div>
            </div>
          </fieldset>
          <fieldset>
            <legend>Verified artifact readiness</legend>
            <p className="field-note">
              Mark only documents available for bank verification. Missing items
              remain visible after screening.
            </p>
            <div className="artifact-checklist">
              {(Object.entries(ARTIFACT_LABELS) as [ArtifactKey, string][]).map(
                ([key, label]) => (
                  <label key={key}>
                    <input
                      type="checkbox"
                      checked={artifacts[key]}
                      onChange={(event) =>
                        setArtifacts((current) => ({
                          ...current,
                          [key]: event.target.checked,
                        }))
                      }
                    />
                    <FileCheck2 size={16} />
                    <span>{label}</span>
                  </label>
                ),
              )}
            </div>
          </fieldset>
        </>
      ) : null}

      <div className="form-navigation">
        {step > 1 ? (
          <button
            type="button"
            className="secondary-action"
            onClick={() => setStep((current) => current - 1)}
          >
            <ArrowLeft size={16} /> Back
          </button>
        ) : (
          <span />
        )}
        {step < 3 ? (
          <button
            type="button"
            className="primary-action step-action"
            disabled={!stepComplete(form, step)}
            onClick={() => setStep((current) => current + 1)}
          >
            Continue <ArrowRight size={16} />
          </button>
        ) : null}
      </div>

      {step === 3 ? (
        <button
          className="primary-action"
          type="submit"
          disabled={!complete || loading}
        >
          <ShieldCheck size={17} />
          {loading ? "Running screening" : "Generate indicative assessment"}
          <ArrowRight size={17} />
        </button>
      ) : null}
      <p className="form-footnote">
        Synthetic pre-screening only. No bureau pull, KYC verification, or
        credit decision.
      </p>
    </form>
  );
}

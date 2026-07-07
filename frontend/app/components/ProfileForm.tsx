"use client";
import { useReducer } from "react";

const SECTORS = [
  { value: "manufacturing", label: "Manufacturing" },
  { value: "services", label: "Services" },
  { value: "textiles", label: "Textiles" },
  { value: "agri-processing", label: "Agri-Processing" },
];

const QUESTIONS = [
  {
    field: "hasLoans" as const,
    label: "Does your business have any existing bank loans or EMIs?",
    hint: "Used to assess repayment behaviour from bank account data",
  },
  {
    field: "highConcentration" as const,
    label: "Does a single customer account for more than 50% of your revenue?",
    hint: "High buyer concentration increases vulnerability under shock scenarios",
  },
  {
    field: "seasonal" as const,
    label: "Does your revenue vary significantly month-to-month by season?",
    hint: "e.g. agri-linked, festival-driven, or harvest-cycle businesses",
  },
];

interface FormState {
  sector: string;
  years: string;
  hasLoans: "yes" | "no" | "";
  highConcentration: "yes" | "no" | "";
  seasonal: "yes" | "no" | "";
}

type Action =
  | { field: "sector" | "years"; value: string }
  | {
      field: "hasLoans" | "highConcentration" | "seasonal";
      value: "yes" | "no";
    };

function reducer(state: FormState, action: Action): FormState {
  return { ...state, [action.field]: action.value };
}

const INITIAL: FormState = {
  sector: "",
  years: "",
  hasLoans: "",
  highConcentration: "",
  seasonal: "",
};

// Derive profile type from form answers — computed at submit, not stored in state
function deriveProfileType(f: FormState): string {
  if (f.hasLoans === "no") return "ntc";
  if (f.highConcentration === "yes") return "buyer_concentrated";
  if (f.seasonal === "yes") return "seasonal";
  return "healthy";
}

function isComplete(f: FormState): boolean {
  return (
    f.sector !== "" &&
    f.years !== "" &&
    f.hasLoans !== "" &&
    f.highConcentration !== "" &&
    f.seasonal !== ""
  );
}

interface Props {
  onSubmit: (
    sector: string,
    yearsOperating: number,
    profileType: string,
  ) => void;
  loading: boolean;
}

export default function ProfileForm({ onSubmit, loading }: Props) {
  const [form, dispatch] = useReducer(reducer, INITIAL);
  const complete = isComplete(form);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!complete || loading) return;
    onSubmit(form.sector, parseInt(form.years, 10), deriveProfileType(form));
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <h2 className="text-base font-semibold text-slate-800">
          Tell us about your business
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          We&apos;ll run a stress-tested credit health analysis tailored to your
          profile.
        </p>
      </div>

      {/* Sector */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-slate-700">
          Business Sector
        </label>
        <select
          id="sector"
          value={form.sector}
          onChange={(e) => dispatch({ field: "sector", value: e.target.value })}
          aria-label="Business Sector"
          className="w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option value="">Select your sector…</option>
          {SECTORS.map((s) => (
            <option key={s.value} value={s.value}>
              {s.label}
            </option>
          ))}
        </select>
      </div>

      {/* Years in business */}
      <div className="space-y-1.5">
        <label className="block text-sm font-medium text-slate-700">
          Years in Business
        </label>
        <input
          type="number"
          min="0"
          max="99"
          placeholder="e.g. 5"
          value={form.years}
          onChange={(e) => dispatch({ field: "years", value: e.target.value })}
          className="w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {/* Yes / No questions */}
      {QUESTIONS.map(({ field, label, hint }) => (
        <div key={field} className="space-y-2">
          <p className="text-sm font-medium text-slate-700">{label}</p>
          <p className="text-xs text-slate-400">{hint}</p>
          <div className="flex gap-3">
            {(["yes", "no"] as const).map((opt) => (
              <label
                key={opt}
                className={`flex flex-1 cursor-pointer items-center justify-center rounded-md border px-4 py-2.5 text-sm font-medium transition-colors ${
                  form[field] === opt
                    ? "border-blue-600 bg-blue-50 text-blue-800"
                    : "border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <input
                  type="radio"
                  name={field}
                  value={opt}
                  checked={form[field] === opt}
                  onChange={() => dispatch({ field, value: opt })}
                  className="sr-only"
                />
                {opt === "yes" ? "Yes" : "No"}
              </label>
            ))}
          </div>
        </div>
      ))}

      <button
        type="submit"
        disabled={!complete || loading}
        className="w-full rounded-md bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? "Running analysis…" : "Analyse My MSME →"}
      </button>
    </form>
  );
}

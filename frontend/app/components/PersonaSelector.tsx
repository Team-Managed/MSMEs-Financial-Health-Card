"use client";
import type { Persona } from "@/lib/types";

interface Props {
  personas: Persona[];
  selected: string | null;
  onSelect: (id: string) => void;
  loading: boolean;
}

const SECTOR_LABELS: Record<string, string> = {
  manufacturing: "Manufacturing",
  services: "Services",
  textiles: "Textiles",
  "agri-processing": "Agri-Processing",
};

export default function PersonaSelector({
  personas,
  selected,
  onSelect,
  loading,
}: Props) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Select MSME Profile
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {personas.map((p) => (
          <button
            key={p.id}
            onClick={() => onSelect(p.id)}
            disabled={loading}
            className={`rounded-md border px-4 py-3 text-left transition-colors ${
              selected === p.id
                ? "border-blue-600 bg-blue-50 text-blue-900"
                : "border-slate-200 bg-white text-slate-700 hover:border-blue-300 hover:bg-slate-50"
            } disabled:opacity-50`}
          >
            <p className="font-medium">{p.business_name}</p>
            <p className="mt-0.5 text-xs text-slate-500">
              {SECTOR_LABELS[p.sector] ?? p.sector}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}

import type { Persona } from "@/lib/types";
import { fetchPersonas } from "@/lib/api";
import Dashboard from "./components/Dashboard";

export default async function Home() {
  let personas: Persona[] = [];
  let initialError: string | null = null;
  try {
    personas = await fetchPersonas();
  } catch {
    initialError = "Could not load personas — is the backend running?";
  }
  return <Dashboard initialPersonas={personas} initialError={initialError} />;
}

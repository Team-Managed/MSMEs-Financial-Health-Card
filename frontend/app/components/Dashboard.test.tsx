import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import Dashboard from "./Dashboard";
import { analyzePersona, fetchPersonas } from "@/lib/api";

jest.mock("@/lib/api", () => ({
    analyzeCustom: jest.fn(),
    analyzePersona: jest.fn(),
    fetchPersonas: jest.fn(),
}));

jest.mock("next/dynamic", () => () => function MockDynamicComponent() {
    return <div />;
});

jest.mock("./ProfileForm", () => function MockProfileForm() {
    return <div />;
});

jest.mock("./PersonaSelector", () => function MockPersonaSelector(props: {
    onSelect: (id: string) => void;
}) {
    return <button onClick={() => props.onSelect("healthy")}>Analyze healthy persona</button>;
});

jest.mock("./HealthCard", () => function MockHealthCard() {
    return <div />;
});
jest.mock("./WeightRationale", () => function MockWeightRationale() {
    return <div />;
});

const mockFetchPersonas = fetchPersonas as jest.MockedFunction<typeof fetchPersonas>;
const mockAnalyzePersona = analyzePersona as jest.MockedFunction<typeof analyzePersona>;

describe("Dashboard", () => {
    it("shows the grounding trace after a persona analysis", async () => {
        mockFetchPersonas.mockResolvedValue([
            { id: "healthy", business_name: "Healthy Co", sector: "services" },
        ]);
        mockAnalyzePersona.mockResolvedValue({
            profile_summary: {},
            cfcr_baseline: 1.3,
            cfcr_by_scenario: [],
            weights_used: { gst: 0.3, upi: 0.3, aa: 0.25, epfo: 0.15 },
            weight_rationale: [],
            baseline_score: 72,
            stress_results: [],
            narrative: "Grounded narrative",
            grounding_trace: [{
                claim: "CFCR is 1.3",
                type: "numeric",
                source: "risk_engine_output",
                status: "pass",
            }],
        });

        render(<Dashboard />);
        fireEvent.click(await screen.findByRole("button", { name: "Analyze healthy persona" }));

        await waitFor(() => expect(mockAnalyzePersona).toHaveBeenCalledWith("healthy"));
        expect(await screen.findByText("Grounding Trace")).toBeInTheDocument();
    });
});
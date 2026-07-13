import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import Dashboard from "./Dashboard";
import { analyzeCustom } from "@/lib/api";

jest.mock("@/lib/api", () => ({
    analyzeCustom: jest.fn(),
}));

jest.mock("next/dynamic", () => () => function MockDynamicComponent() {
    return <div />;
});

jest.mock("./ProfileForm", () => {
    return {
        __esModule: true,
        ARTIFACT_LABELS: {
            bankStatements: "Bank statements",
            gstItr: "GST and ITR",
            promoterKyc: "Promoter KYC",
            businessIdentity: "Business identity",
            financialStatements: "Financial statements",
        },
        default: function MockProfileForm({ onSubmit }: { onSubmit: (params: unknown) => void }) {
            return (
                <button
                    onClick={() => onSubmit({
                        sector: "services",
                        yearsOperating: 5,
                        profileType: "healthy",
                        msmeTier: "micro",
                        gstRegistered: true,
                        employeeTier: "micro",
                        requestedAmountLakh: 10,
                        annualInterestRatePct: 12,
                        expectedUtilizationPct: 75,
                        annualTurnoverLakh: 100,
                        avgMonthlyInflowLakh: 10,
                        avgMonthlyOperatingOutflowLakh: 7,
                        avgBankBalanceLakh: 8,
                        existingMonthlyEmiLakh: 1,
                        topBuyerSharePct: 20,
                        bouncedPayments12mo: 0,
                        gstFilingConsistencyPct: 95,
                        yoyGrowthPct: 12,
                        verifiedArtifacts: {
                            bankStatements: true,
                            gstItr: true,
                            promoterKyc: true,
                            businessIdentity: true,
                            financialStatements: true,
                        },
                    })}
                >
                    Run mocked analysis
                </button>
            );
        },
    };
});

jest.mock("./HealthCard", () => function MockHealthCard() {
    return <div />;
});
jest.mock("./WeightRationale", () => function MockWeightRationale() {
    return <div />;
});

const mockAnalyzeCustom = analyzeCustom as jest.MockedFunction<typeof analyzeCustom>;

describe("Dashboard", () => {
    it("shows the grounding trace after an artifact-verified analysis", async () => {
        mockAnalyzeCustom.mockResolvedValue({
            profile_summary: {},
            cfcr_baseline: 1.3,
            cfcr_by_scenario: [],
            weights_used: { gst: 0.3, upi: 0.3, aa: 0.25, epfo: 0.15 },
            weight_rationale: [],
            baseline_score: 72,
            stress_results: [],
            tail_risk: {
                probability_cfcr_below_one: 0.1,
                cfcr_p05: 0.9,
                expected_shortfall: 0.08,
                simulations: 5000,
                model_version: "borrower_cashflow_v1",
                assumptions: ["Sensitivity estimate"],
            },
            narrative: "Grounded narrative",
            grounding_trace: [{
                claim: "CFCR is 1.3",
                type: "numeric",
                source: "risk_engine_output",
                status: "pass",
            }],
        });

        render(<Dashboard />);
        fireEvent.click(await screen.findByRole("button", { name: "Run mocked analysis" }));

        await waitFor(() => expect(mockAnalyzeCustom).toHaveBeenCalled());
        expect(await screen.findByText("Grounding trace")).toBeInTheDocument();
    });
});
import { analyzeCustom, fetchPersonas, analyzePersona } from "./api";

global.fetch = jest.fn();

const mockFetch = global.fetch as jest.Mock;

describe("fetchPersonas", () => {
  it("calls /api/personas and returns array", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [
        { id: "healthy", business_name: "Test Co", sector: "manufacturing" },
      ],
    });
    const result = await fetchPersonas();
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("healthy");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/personas",
    );
  });
});

describe("analyzePersona", () => {
  it("posts to /api/msme/{id}/analyze and returns AnalysisResponse", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        cfcr_baseline: 1.3,
        baseline_score: 72.0,
        cfcr_by_scenario: [],
        weights_used: { gst: 0.3, upi: 0.3, aa: 0.25, epfo: 0.15 },
        weight_rationale: [],
        stress_results: [],
        narrative: "ok",
        grounding_trace: [],
        profile_summary: {},
      }),
    });
    const result = await analyzePersona("healthy");
    expect(result.cfcr_baseline).toBe(1.3);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/msme/healthy/analyze",
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("analyzeCustom", () => {
  it("includes GST registration in the custom analysis request", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ cfcr_baseline: 1.3 }),
    });

    await analyzeCustom({
      sector: "services",
      yearsOperating: 5,
      profileType: "healthy",
      msmeTier: "micro",
      gstRegistered: false,
      employeeTier: "micro",
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/analyze",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining('"gst_registered":false'),
      }),
    );
  });
});

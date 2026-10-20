/**
 * Tests for the API client.
 *
 * Strategy: mock the axios module entirely; `axios.create` returns a mock
 * instance whose `interceptors.*.use` capture the handlers registered by
 * client.ts. Tests then invoke those handlers directly (mirroring what the
 * real axios pipeline does) and program the HTTP methods per test.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const m = vi.hoisted(() => {
  const requestUse = vi.fn();
  const responseUse = vi.fn();
  const get = vi.fn();
  const post = vi.fn();
  const del = vi.fn();
  const create = vi.fn((..._args: unknown[]) => ({
    interceptors: {
      request: { use: requestUse },
      response: { use: responseUse },
    },
    get,
    post,
    delete: del,
  }));
  return { requestUse, responseUse, get, post, del, create };
});

vi.mock("axios", () => ({ default: { create: m.create } }));

import {
  ApiKeyError,
  ValidationError,
  createAnalysis,
  getApiKey,
  setApiKey,
} from "./client";

type ReqConfig = { url?: string; headers: Record<string, unknown> };

function requestInterceptor(): (config: ReqConfig) => ReqConfig {
  expect(m.requestUse).toHaveBeenCalled();
  return m.requestUse.mock.calls[0][0] as (config: ReqConfig) => ReqConfig;
}

function responseErrorInterceptor(): (error: unknown) => Promise<unknown> {
  expect(m.responseUse).toHaveBeenCalled();
  return m.responseUse.mock.calls[0][1] as (error: unknown) => Promise<unknown>;
}

beforeEach(() => {
  // Only reset the HTTP-method mocks — the interceptor registration calls
  // happened at module import and must survive across tests.
  m.get.mockReset();
  m.post.mockReset();
  m.del.mockReset();
  localStorage.clear();
});

describe("API key storage", () => {
  it("setApiKey persists to localStorage and getApiKey reads it back", () => {
    expect(getApiKey()).toBe("");
    setApiKey("test-key");
    expect(localStorage.getItem("neurosight.apiKey")).toBe("test-key");
    expect(getApiKey()).toBe("test-key");
  });
});

describe("API key storage when localStorage is unavailable", () => {
  it("setApiKey keeps the key in memory when localStorage.setItem throws", () => {
    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("storage denied", "SecurityError");
      });

    try {
      setApiKey("memory-key");

      // Nothing was persisted...
      expect(localStorage.getItem("neurosight.apiKey")).toBeNull();
      // ...but the key is still served for this session.
      expect(getApiKey()).toBe("memory-key");
    } finally {
      setItemSpy.mockRestore();
    }
  });

  it("getApiKey falls back to the in-memory key when localStorage.getItem throws", () => {
    setApiKey("memory-key");
    const getItemSpy = vi
      .spyOn(Storage.prototype, "getItem")
      .mockImplementation(() => {
        throw new DOMException("storage denied", "SecurityError");
      });

    try {
      expect(getApiKey()).toBe("memory-key");
    } finally {
      getItemSpy.mockRestore();
    }
  });
});

describe("request interceptor", () => {
  it("attaches X-API-Key for /api/ urls", () => {
    setApiKey("test-key");
    const interceptor = requestInterceptor();
    const config = interceptor({ url: "/api/v1/analyses", headers: {} });
    expect(config.headers["X-API-Key"]).toBe("test-key");
  });

  it("does NOT attach X-API-Key for /health", () => {
    setApiKey("test-key");
    const interceptor = requestInterceptor();
    const config = interceptor({ url: "/health", headers: {} });
    expect(config.headers["X-API-Key"]).toBeUndefined();
  });

  it("creates the axios instance with an empty-string baseURL fallback", () => {
    expect(m.create).toHaveBeenCalledTimes(1);
    const [options] = m.create.mock.calls[0];
    expect(options).toEqual(expect.objectContaining({ baseURL: expect.any(String) }));
  });
});

describe("response error interceptor", () => {
  it("converts a 401 into ApiKeyError and dispatches the unauthorized event", async () => {
    const listener = vi.fn();
    window.addEventListener("neurosight:unauthorized", listener);
    const interceptor = responseErrorInterceptor();
    const error401 = { response: { status: 401, data: { detail: "bad key" } } };

    await expect(interceptor(error401)).rejects.toBeInstanceOf(ApiKeyError);

    expect(listener).toHaveBeenCalledTimes(1);
    window.removeEventListener("neurosight:unauthorized", listener);
  });

  it("converts a 422 with {detail:{problems}} into ValidationError carrying problems", async () => {
    const interceptor = responseErrorInterceptor();
    const error422 = {
      response: { status: 422, data: { detail: { problems: ["a", "b"] } } },
    };

    let caught: unknown;
    try {
      await interceptor(error422);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ValidationError);
    expect((caught as ValidationError).problems).toEqual(["a", "b"]);
  });

  it("converts a 429 rate limit error into ValidationError", async () => {
    const interceptor = responseErrorInterceptor();
    const error429 = {
      response: {
        status: 429,
        data: { detail: "Too many analyses — retry shortly." },
      },
    };

    let caught: unknown;
    try {
      await interceptor(error429);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ValidationError);
    expect((caught as ValidationError).problems).toEqual([
      "Too many analyses — retry shortly.",
    ]);
  });

  it("leaves other errors untouched (pass-through)", async () => {
    const interceptor = responseErrorInterceptor();
    const error500 = { response: { status: 500, data: { detail: "boom" } } };
    await expect(interceptor(error500)).rejects.toBe(error500);
  });
});

describe("createAnalysis", () => {
  it("builds FormData: files under 'files', visit_labels_json, include_gradcam 'true'", async () => {
    m.post.mockResolvedValueOnce({
      data: {
        analysis_id: "an-1",
        status: "queued",
        visit_count: 2,
        visit_labels: ["Baseline", "Follow-up"],
        poll_url: "/api/v1/analyses/an-1",
      },
    });

    const files = [
      new File(["a"], "a.nii.gz"),
      new File(["b"], "b.nii.gz"),
    ];
    const response = await createAnalysis({
      files,
      visitLabels: ["Baseline", "Follow-up"],
      includeGradcam: true,
    });

    expect(response.analysis_id).toBe("an-1");
    const [url, form] = m.post.mock.calls[0];
    expect(url).toBe("/api/v1/analyses");
    expect(form).toBeInstanceOf(FormData);

    const sentFiles = (form as FormData).getAll("files") as File[];
    expect(sentFiles).toHaveLength(2);
    expect(sentFiles.map((f) => f.name)).toEqual(["a.nii.gz", "b.nii.gz"]);
    expect((form as FormData).get("visit_labels_json")).toBe(
      JSON.stringify(["Baseline", "Follow-up"]),
    );
    expect((form as FormData).get("include_gradcam")).toBe("true");
  });

  it("sends include_gradcam 'false' and optional subject_ref/demographics as JSON strings", async () => {
    m.post.mockResolvedValueOnce({ data: { analysis_id: "an-2", status: "queued", visit_count: 1, visit_labels: ["S1"], poll_url: "/api/v1/analyses/an-2" } });

    await createAnalysis({
      files: [new File(["x"], "scan.nii.gz")],
      visitLabels: ["S1"],
      subjectRef: "P-01",
      demographics: { sex: "M", age: 74 },
      includeGradcam: false,
    });

    const form = m.post.mock.calls[0][1] as FormData;
    expect(form.get("include_gradcam")).toBe("false");
    expect(form.get("subject_ref")).toBe("P-01");
    expect(form.get("demographics")).toBe(JSON.stringify({ sex: "M", age: 74 }));
  });

  it("omits subject_ref when not provided", async () => {
    m.post.mockResolvedValueOnce({ data: { analysis_id: "an-3", status: "queued", visit_count: 1, visit_labels: ["S1"], poll_url: "/api/v1/analyses/an-3" } });

    await createAnalysis({
      files: [new File(["x"], "scan.nii.gz")],
      visitLabels: ["S1"],
      includeGradcam: false,
    });

    const form = m.post.mock.calls[0][1] as FormData;
    expect(form.get("subject_ref")).toBeNull();
    expect(form.get("demographics")).toBeNull();
  });
});

describe("typed GET helpers", () => {
  it("getHealth requests /health and unwraps data", async () => {
    const { getHealth } = await import("./client");
    const payload = { status: "ok", model_version: "v1", uptime_s: 12, device: "cpu" };
    m.get.mockResolvedValueOnce({ data: payload });
    await expect(getHealth()).resolves.toEqual(payload);
    expect(m.get).toHaveBeenCalledWith("/health");
  });

  it("getAnalysis requests the single-analysis endpoint", async () => {
    const { getAnalysis } = await import("./client");
    m.get.mockResolvedValueOnce({ data: { analysis_id: "abc", status: "queued" } });
    await expect(getAnalysis("abc")).resolves.toEqual({ analysis_id: "abc", status: "queued" });
    expect(m.get).toHaveBeenCalledWith("/api/v1/analyses/abc");
  });

  it("getPatients unwraps the patients array", async () => {
    const { getPatients } = await import("./client");
    m.get.mockResolvedValueOnce({ data: { patients: ["P-01", "P-02"] } });
    await expect(getPatients()).resolves.toEqual(["P-01", "P-02"]);
    expect(m.get).toHaveBeenCalledWith("/api/v1/patients");
  });

  it("getPatientHistory and deleteAnalysis hit the right URLs", async () => {
    const { getPatientHistory, deleteAnalysis } = await import("./client");
    m.get.mockResolvedValueOnce({ data: [] });
    m.del.mockResolvedValueOnce({ data: { analysis_id: "abc", status: "cancelled" } });

    await expect(getPatientHistory("P 01")).resolves.toEqual([]);
    expect(m.get).toHaveBeenCalledWith("/api/v1/patients/P%2001/history");

    await expect(deleteAnalysis("abc")).resolves.toEqual({ analysis_id: "abc", status: "cancelled" });
    expect(m.del).toHaveBeenCalledWith("/api/v1/analyses/abc");
  });
});

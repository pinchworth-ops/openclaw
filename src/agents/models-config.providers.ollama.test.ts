import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveImplicitProviders, resolveOllamaApiBase } from "./models-config.providers.js";

describe("resolveOllamaApiBase", () => {
  it("returns default localhost base when no configured URL is provided", () => {
    expect(resolveOllamaApiBase()).toBe("http://127.0.0.1:11434");
  });

  it("strips /v1 suffix from OpenAI-compatible URLs", () => {
    expect(resolveOllamaApiBase("http://ollama-host:11434/v1")).toBe("http://ollama-host:11434");
    expect(resolveOllamaApiBase("http://ollama-host:11434/V1")).toBe("http://ollama-host:11434");
  });

  it("keeps URLs without /v1 unchanged", () => {
    expect(resolveOllamaApiBase("http://ollama-host:11434")).toBe("http://ollama-host:11434");
  });

  it("handles trailing slash before canonicalizing", () => {
    expect(resolveOllamaApiBase("http://ollama-host:11434/v1/")).toBe("http://ollama-host:11434");
    expect(resolveOllamaApiBase("http://ollama-host:11434/")).toBe("http://ollama-host:11434");
  });
});

describe("Ollama provider", () => {
  it("should not include ollama when no API key is configured", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({ agentDir });

    expect(providers?.ollama).toBeUndefined();
  });

  it("should use native ollama api type", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    process.env.OLLAMA_API_KEY = "test-key";

    try {
      const providers = await resolveImplicitProviders({ agentDir });

      expect(providers?.ollama).toBeDefined();
      expect(providers?.ollama?.apiKey).toBe("OLLAMA_API_KEY");
      expect(providers?.ollama?.api).toBe("ollama");
      expect(providers?.ollama?.baseUrl).toBe("http://127.0.0.1:11434");
    } finally {
      delete process.env.OLLAMA_API_KEY;
    }
  });

  it("should preserve explicit ollama baseUrl on implicit provider injection", async () => {
    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    process.env.OLLAMA_API_KEY = "test-key";

    try {
      const providers = await resolveImplicitProviders({
        agentDir,
        explicitProviders: {
          ollama: {
            baseUrl: "http://192.168.20.14:11434/v1",
            api: "openai-completions",
            models: [],
          },
        },
      });

      // Native API strips /v1 suffix via resolveOllamaApiBase()
      expect(providers?.ollama?.baseUrl).toBe("http://192.168.20.14:11434");
    } finally {
      delete process.env.OLLAMA_API_KEY;
    }
  });

  it("should have correct model structure without streaming override", () => {
    const mockOllamaModel = {
      id: "llama3.3:latest",
      name: "llama3.3:latest",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 8192,
    };

    // Native Ollama provider does not need streaming: false workaround
    expect(mockOllamaModel).not.toHaveProperty("params");
  });
});

// Regression tests for #8663 and #11283 — requires temporarily unblocking the
// VITEST env guard inside discoverOllamaModels so fetch calls are exercised.
describe("Ollama discovery with remote baseUrl", () => {
  let savedVitest: string | undefined;
  let savedNodeEnv: string | undefined;
  let savedFetch: typeof globalThis.fetch;

  afterEach(() => {
    // Restore env vars and fetch
    if (savedVitest !== undefined) {
      process.env.VITEST = savedVitest;
    } else {
      delete process.env.VITEST;
    }
    if (savedNodeEnv !== undefined) {
      process.env.NODE_ENV = savedNodeEnv;
    } else {
      delete process.env.NODE_ENV;
    }
    globalThis.fetch = savedFetch;
    delete process.env.OLLAMA_API_KEY;
  });

  function enableDiscovery() {
    savedVitest = process.env.VITEST;
    savedNodeEnv = process.env.NODE_ENV;
    savedFetch = globalThis.fetch;
    delete process.env.VITEST;
    delete process.env.NODE_ENV;
  }

  // Regression test for #8663: discovery must hit the configured remote host,
  // NOT the hardcoded 'http://127.0.0.1:11434', when baseUrl is set.
  it("uses configured remote baseUrl for /api/tags discovery, not hardcoded localhost (fixes #8663)", async () => {
    enableDiscovery();
    process.env.OLLAMA_API_KEY = "test-key";
    const capturedUrls: string[] = [];

    globalThis.fetch = vi.fn().mockImplementation(async (url: string | URL) => {
      capturedUrls.push(String(url));
      if (String(url).includes("/api/tags")) {
        return {
          ok: true,
          json: async () => ({ models: [{ name: "llama3.3:latest" }] }),
        } as Response;
      }
      return { ok: false, status: 404 } as Response;
    }) as typeof fetch;

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        ollama: {
          baseUrl: "http://remote-ollama:11434/v1",
          api: "ollama",
          models: [],
        },
      },
    });

    // The /api/tags fetch must target the configured remote host, NOT 127.0.0.1
    const tagsCall = capturedUrls.find((u) => u.includes("/api/tags"));
    expect(tagsCall).toBeDefined();
    // The /v1 suffix must be stripped before appending /api/tags (native API path)
    expect(tagsCall).toBe("http://remote-ollama:11434/api/tags");
    expect(tagsCall).not.toContain("127.0.0.1");

    // The returned provider must reflect the configured (v1-stripped) baseUrl
    expect(providers?.ollama?.baseUrl).toBe("http://remote-ollama:11434");
    expect(providers?.ollama?.models).toHaveLength(1);
    expect(providers?.ollama?.models?.[0]?.id).toBe("llama3.3:latest");
  });

  // Regression test for #11283: the provider must use api:"ollama" (HTTP-based),
  // ensuring attempt.ts routes through createOllamaStreamFn and never invokes
  // the local 'ollama' CLI binary.
  it("registers provider with api:ollama so HTTP streaming is used, not CLI binary (fixes #11283)", async () => {
    enableDiscovery();
    process.env.OLLAMA_API_KEY = "test-key";

    globalThis.fetch = vi.fn().mockImplementation(async (url: string | URL) => {
      if (String(url).includes("/api/tags")) {
        return {
          ok: true,
          json: async () => ({ models: [{ name: "mistral:latest" }] }),
        } as Response;
      }
      return { ok: false, status: 404 } as Response;
    }) as typeof fetch;

    const agentDir = mkdtempSync(join(tmpdir(), "openclaw-test-"));
    const providers = await resolveImplicitProviders({
      agentDir,
      explicitProviders: {
        ollama: {
          baseUrl: "http://remote-ollama:11434/v1",
          api: "ollama",
          models: [],
        },
      },
    });

    // Provider must carry api:"ollama" so that attempt.ts routes through
    // createOllamaStreamFn (HTTP /api/chat) and never invokes the ollama CLI binary.
    expect(providers?.ollama?.api).toBe("ollama");
  });
});

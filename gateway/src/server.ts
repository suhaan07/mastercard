/**
 * Fastify gateway: the network hop between a merchant and the scorer.
 *
 * This exists for two reasons that a single-process demo cannot provide.
 *
 * First, the latency budget becomes an honest measurement. The design doc puts
 * auth-time scoring at p99 under ~50 ms inside a round trip of a few hundred.
 * Measuring that in-process measures a function call; measuring it across a
 * real hop measures something a payments engineer would recognise.
 *
 * Second, and more important: it makes the merchant-view/network-view toggle
 * *physically* true. A merchant service is a separate process that has only
 * ever been handed its own institution's traffic. When the console flips to
 * merchant view, the missed detections are missed because the evidence was
 * genuinely not in that process -- not because a filter hid it. That is the
 * difference between demonstrating the "why Mastercard" claim and asserting it.
 */

import Fastify from "fastify";
import cors from "@fastify/cors";

const SCORER_URL = process.env.SCORER_URL ?? "http://127.0.0.1:8000";
const PORT = Number(process.env.GATEWAY_PORT ?? 8080);

type ViewScope = "merchant" | "issuer" | "network";

interface ScoreResponse {
  event_id: string;
  score: number;
  band: string;
  action: string;
  reason_codes: string[];
  ring_id: string | null;
  view: ViewScope;
  propagated: boolean;
  latency_ms: number;
}

const app = Fastify({ logger: { level: process.env.LOG_LEVEL ?? "warn" } });
await app.register(cors, { origin: true });

/** Round-trip latencies observed at the gateway, for the /metrics endpoint. */
const latencies: number[] = [];

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

async function callScorer(path: string, body: unknown, view: ViewScope): Promise<ScoreResponse> {
  const res = await fetch(`${SCORER_URL}${path}?view=${view}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`scorer ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as ScoreResponse;
}

app.get("/health", async () => {
  let scorer = "unreachable";
  try {
    const r = await fetch(`${SCORER_URL}/health`);
    scorer = r.ok ? "ok" : `http ${r.status}`;
  } catch {
    scorer = "unreachable";
  }
  return { status: "ok", scorer, scorer_url: SCORER_URL };
});

/**
 * Authorise a transaction end to end.
 *
 * The merchant sends an auth request; the gateway scores it and translates the
 * decision into an actual authorisation outcome. Note that HONEYPOT does not
 * decline: it returns a plausible result, so the attacker cannot tell a
 * poisoned answer from a real one and their whole harvest becomes suspect.
 */
app.post<{ Body: Record<string, unknown>; Querystring: { view?: ViewScope } }>(
  "/authorize",
  async (request, reply) => {
    const view = request.query.view ?? "network";
    const started = performance.now();

    let decision: ScoreResponse;
    try {
      decision = await callScorer("/score/auth", request.body, view);
    } catch (err) {
      // Fail open on a scorer outage. Declining every transaction because the
      // risk service is down is a far worse outcome than missing fraud for a
      // few seconds -- an availability failure must not become an outage for
      // every legitimate cardholder.
      request.log.error({ err }, "scorer unavailable; failing open");
      return reply.send({
        approved: true,
        action: "allow",
        degraded: true,
        reason: "risk service unavailable, failed open",
      });
    }

    const elapsed = performance.now() - started;
    latencies.push(elapsed);
    if (latencies.length > 20000) latencies.splice(0, 10000);

    let approved: boolean;
    let responseCode: string;
    switch (decision.action) {
      case "allow":
        approved = true;
        responseCode = "approved";
        break;
      case "step_up":
        approved = false;
        responseCode = "challenge_required";
        break;
      case "throttle":
        approved = false;
        responseCode = "declined_try_again_later";
        break;
      case "honeypot": {
        const hp = await fetch(
          `${SCORER_URL}/honeypot?card_token=${encodeURIComponent(
            String((request.body as Record<string, unknown>).card_token ?? "")
          )}`,
          { method: "POST" }
        );
        const body = (await hp.json()) as { response_code: string };
        approved = body.response_code === "approved";
        responseCode = body.response_code;
        break;
      }
      default:
        approved = false;
        responseCode = "declined_risk";
    }

    return reply.send({
      approved,
      response_code: responseCode,
      action: decision.action,
      band: decision.band,
      score: decision.score,
      ring_id: decision.ring_id,
      propagated: decision.propagated,
      reason_codes: decision.reason_codes,
      view,
      scorer_latency_ms: decision.latency_ms,
      gateway_latency_ms: elapsed,
    });
  }
);

app.post<{ Body: Record<string, unknown>; Querystring: { view?: ViewScope } }>(
  "/onboard",
  async (request, reply) => {
    const view = request.query.view ?? "network";
    const decision = await callScorer("/score/onboarding", request.body, view);
    return reply.send(decision);
  }
);

app.get("/metrics", async () => ({
  gateway: {
    n: latencies.length,
    p50_ms: percentile(latencies, 50),
    p95_ms: percentile(latencies, 95),
    p99_ms: percentile(latencies, 99),
  },
}));

/** Proxy the console's read-only queries so the UI talks to one origin. */
for (const path of ["/communities", "/metrics/scorer", "/health/scorer"]) {
  const target = path.replace("/scorer", "").replace("/health", "/health");
  app.get<{ Querystring: Record<string, string> }>(path, async (request) => {
    // Forward the query string: the console asks for a specific number of
    // communities, and dropping it silently truncated the ring list.
    const qs = new URLSearchParams(request.query as Record<string, string>).toString();
    const r = await fetch(`${SCORER_URL}${target}${qs ? `?${qs}` : ""}`);
    return await r.json();
  });
}

app.get<{ Params: { eventId: string } }>("/explain/:eventId", async (request, reply) => {
  const r = await fetch(`${SCORER_URL}/explain/${request.params.eventId}`);
  return reply.code(r.status).send(await r.json());
});

/**
 * Server-sent events, streamed through rather than buffered.
 *
 * Reading the whole body before replying would defeat the point -- the console
 * is meant to watch decisions arrive at the rate they are made. Fastify is
 * handed the upstream body as a stream so backpressure stays end to end.
 */
app.get<{ Querystring: Record<string, string> }>("/stream", async (request, reply) => {
  const qs = new URLSearchParams(request.query as Record<string, string>).toString();
  const upstream = await fetch(`${SCORER_URL}/stream${qs ? `?${qs}` : ""}`);
  if (!upstream.ok || upstream.body === null) {
    return reply.code(upstream.status || 502).send({ error: "scorer stream unavailable" });
  }
  reply.raw.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const reader = upstream.body.getReader();
  request.raw.on("close", () => void reader.cancel().catch(() => {}));
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      reply.raw.write(value);
    }
  } catch {
    // Client went away mid-replay; nothing to report.
  } finally {
    reply.raw.end();
  }
});

app.get<{ Params: { ringId: string } }>("/graph/ring/:ringId", async (request) => {
  const r = await fetch(`${SCORER_URL}/graph/ring/${request.params.ringId}`);
  return await r.json();
});

app.post<{ Params: { ringId: string }; Querystring: Record<string, string> }>(
  "/narrate/:ringId",
  async (request) => {
    const qs = new URLSearchParams(request.query as Record<string, string>).toString();
    const r = await fetch(
      `${SCORER_URL}/narrate/${request.params.ringId}${qs ? `?${qs}` : ""}`,
      { method: "POST" }
    );
    return await r.json();
  }
);

app.post<{ Params: { identityId: string } }>("/confirm/:identityId", async (request) => {
  const r = await fetch(`${SCORER_URL}/confirm/${request.params.identityId}`, {
    method: "POST",
  });
  return await r.json();
});

await app.listen({ port: PORT, host: "0.0.0.0" });
console.log(`[gateway] listening on :${PORT}, scorer at ${SCORER_URL}`);

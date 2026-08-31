/**
 * Mock merchant and issuer services.
 *
 * Each instance is started with one institution id and is handed only that
 * institution's traffic. It genuinely cannot see anyone else's, because nobody
 * ever sends it anyone else's -- which is the point. The merchant-view number in
 * the metrics report is what a process in this position can detect, and the gap
 * to the network view is the argument for network-level deployment.
 *
 * Run:
 *   node dist/institution.js --role merchant --institution inst_00 --port 8101
 *   node dist/institution.js --role issuer   --institution inst_01 --port 8102
 */

import Fastify from "fastify";
import cors from "@fastify/cors";

type Role = "merchant" | "issuer";

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const ROLE = arg("role", "merchant") as Role;
const INSTITUTION = arg("institution", "inst_00");
const PORT = Number(arg("port", ROLE === "merchant" ? "8101" : "8102"));
const GATEWAY_URL = process.env.GATEWAY_URL ?? "http://127.0.0.1:8080";

const app = Fastify({ logger: { level: process.env.LOG_LEVEL ?? "warn" } });
await app.register(cors, { origin: true });

/**
 * Everything this institution has ever observed. A plain in-process store,
 * deliberately: the isolation being demonstrated is that this object only ever
 * contains one institution's events.
 */
const observed: Array<Record<string, unknown>> = [];
const seenDevices = new Set<string>();
const seenAccounts = new Set<string>();

app.get("/health", async () => ({
  status: "ok",
  role: ROLE,
  institution: INSTITUTION,
  observed_events: observed.length,
  distinct_devices: seenDevices.size,
  distinct_accounts: seenAccounts.size,
}));

app.post<{ Body: Record<string, unknown> }>("/authorize", async (request, reply) => {
  const event = request.body;

  // Refuse traffic that is not ours. This is what makes the isolation real
  // rather than a convention: a misrouted event is an error, not a silent
  // widening of what this institution can see.
  if (event.institution_id && event.institution_id !== INSTITUTION) {
    return reply.status(400).send({
      error: "wrong institution",
      expected: INSTITUTION,
      got: event.institution_id,
    });
  }

  observed.push(event);
  if (typeof event.device_id === "string") seenDevices.add(event.device_id);
  if (typeof event.account_id === "string") seenAccounts.add(event.account_id);

  // Scored under the merchant view: only this institution's evidence.
  const res = await fetch(`${GATEWAY_URL}/authorize?view=merchant`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(event),
  });
  const decision = await res.json();
  return reply.send({ institution: INSTITUTION, role: ROLE, ...(decision as object) });
});

/**
 * What this institution can see about a device on its own.
 *
 * Card testing is deliberately sprayed across many merchants and issuers, so
 * each one sees three attempts here and five there -- nothing alarming. This
 * endpoint returns that sliver, and the console puts it beside the network
 * view to show the same device's full pattern.
 */
app.get<{ Params: { deviceId: string } }>("/device/:deviceId", async (request) => {
  const hits = observed.filter((e) => e.device_id === request.params.deviceId);
  const accounts = new Set(hits.map((e) => String(e.account_id)));
  return {
    institution: INSTITUTION,
    device_id: request.params.deviceId,
    attempts_seen_here: hits.length,
    accounts_seen_here: accounts.size,
    note: "this is one institution's sliver; the network sees the rest",
  };
});

await app.listen({ port: PORT, host: "0.0.0.0" });
console.log(`[${ROLE}:${INSTITUTION}] listening on :${PORT}`);

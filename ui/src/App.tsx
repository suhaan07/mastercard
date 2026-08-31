/**
 * Analyst console.
 *
 * Two things on this screen carry the pitch, and everything else is supporting
 * material:
 *
 *  1. The ring graph, and what happens when one account is confirmed: its
 *     siblings light up, including accounts that have never transacted.
 *  2. The merchant/network toggle, which changes what evidence the detector is
 *     given rather than what the chart draws.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ViewScope = "merchant" | "network";

interface Community {
  ring_id: string;
  size: number;
  cohesion: number;
  suspicion: number;
  score: number;
  cross_institution: boolean;
  institutions: string[];
}

interface GraphNode {
  id: string;
  type: string;
  key: string;
  is_member: boolean;
  propagated: boolean;
  institutions: string[];
}

interface GraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

interface RingGraph {
  ring_id: string;
  size: number;
  cohesion: number;
  suspicion: number;
  score: number;
  cross_institution: boolean;
  institutions: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface FlaggedIdentity {
  identity_id: string;
  score: number;
  hops: number;
  dormant: boolean;
  path: { from_node: string; to_node: string; via: string; contribution: number }[];
}

interface ConfirmResult {
  seed: string;
  n_flagged: number;
  n_dormant_flagged: number;
  flagged: FlaggedIdentity[];
}

interface StreamItem {
  event: {
    event_id: string;
    ts: string;
    account_id: string;
    identity_id: string;
    merchant_id: string;
    institution_id: string;
    amount: number;
    approved: boolean;
  };
  decision: {
    event_id: string;
    score: number;
    band: string;
    action: string;
    reason_codes: string[];
    propagated: boolean;
    latency_ms: number;
  };
  scope: { view: string; institution_id: string | null };
}

interface Narrative {
  ring_id: string;
  narrative: string;
  source: "model" | "template";
  model: string | null;
  note: string;
}

interface Explanation {
  event_id: string;
  score: number;
  band: string;
  action: string;
  propagated: boolean;
  reasons: string[];
  evidence_path: { from_node: string; to_node: string; via: string; contribution: number }[];
}

/** Anything that is not "allow" is an intervention; colour by severity. */
const ACTION_COLOR: Record<string, string> = {
  allow: "text-slate-500",
  step_up: "text-sky-300",
  throttle: "text-amber-300",
  honeypot: "text-fuchsia-300",
  block: "text-rose-300",
};

const ACTION_ORDER = ["allow", "step_up", "throttle", "honeypot", "block"];

const NODE_COLOR: Record<string, string> = {
  identity: "#7aa2f7",
  account: "#9ece6a",
  device: "#f7768e",
  ip_asn: "#e0af68",
  address: "#bb9af7",
  phone: "#7dcfff",
  email: "#73daca",
  card_token: "#565f89",
  merchant: "#3b4261",
};

const api = (path: string) => `/api${path}`;

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums text-slate-100">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<ViewScope>("network");
  const [communities, setCommunities] = useState<Community[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [graph, setGraph] = useState<RingGraph | null>(null);
  const [confirmed, setConfirmed] = useState<ConfirmResult | null>(null);
  const [status, setStatus] = useState<string>("connecting");
  const [busy, setBusy] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [items, setItems] = useState<StreamItem[]>([]);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [narrative, setNarrative] = useState<Narrative | null>(null);
  const graphRef = useRef<any>(null);

  /**
   * The replay is scoped by the same toggle as everything else, and the scope
   * is applied in the scorer, not here: in merchant view the stream is one
   * institution's traffic and the rolling windows are built from that alone.
   * Flipping the toggle therefore restarts the replay rather than filtering it.
   */
  useEffect(() => {
    if (!streaming) return;
    setItems([]);
    const es = new EventSource(api(`/stream?speed=60&limit=4000&view=${view}`));
    es.onmessage = (msg) => {
      const item: StreamItem = JSON.parse(msg.data);
      setItems((prev) => {
        const next = [item, ...prev];
        // Bounded: the console shows a live tail, and the report is where
        // whole-run numbers come from.
        return next.length > 400 ? next.slice(0, 400) : next;
      });
    };
    es.onerror = () => {
      es.close();
      setStreaming(false);
    };
    return () => es.close();
  }, [streaming, view]);

  /**
   * The case narrative. Off the auth path by construction -- this is a
   * request an analyst makes about a case that has already been decided,
   * not part of deciding it.
   */
  const narrateRing = useCallback(async () => {
    if (!graph) return;
    setBusy(true);
    try {
      const res = await fetch(api(`/narrate/${graph.ring_id}`), { method: "POST" });
      setNarrative(await res.json());
    } finally {
      setBusy(false);
    }
  }, [graph]);

  const explain = useCallback((eventId: string) => {
    fetch(api(`/explain/${eventId}`))
      .then((r) => (r.ok ? r.json() : null))
      .then(setExplanation)
      .catch(() => setExplanation(null));
  }, []);

  const actionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const it of items) counts[it.decision.action] = (counts[it.decision.action] ?? 0) + 1;
    return ACTION_ORDER.filter((a) => counts[a]).map((a) => ({ action: a, count: counts[a] }));
  }, [items]);

  const interventionRate = items.length
    ? items.filter((i) => i.decision.action !== "allow").length / items.length
    : 0;
  const p99Latency = useMemo(() => {
    if (!items.length) return 0;
    const sorted = items.map((i) => i.decision.latency_ms).sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(0.99 * sorted.length))];
  }, [items]);

  useEffect(() => {
    fetch(api("/health"))
      .then((r) => r.json())
      .then((h) => setStatus(h.scorer === "ok" ? "connected" : `scorer ${h.scorer}`))
      .catch(() => setStatus("gateway unreachable"));

    fetch(api("/communities?top=25"))
      .then((r) => r.json())
      .then(setCommunities)
      .catch(() => setCommunities([]));
  }, []);

  const loadRing = useCallback((ringId: string) => {
    setSelected(ringId);
    setConfirmed(null);
    setNarrative(null);
    setBusy(true);
    fetch(api(`/graph/ring/${ringId}`))
      .then((r) => r.json())
      .then(setGraph)
      .finally(() => setBusy(false));
  }, []);

  /**
   * The demo moment. Confirm one account and let the evidence propagate; the
   * response tells us which siblings were raised and how many of them have
   * never transacted at all.
   */
  const confirmOne = useCallback(async () => {
    if (!graph) return;
    const seed = graph.nodes.find((n) => n.type === "identity" && n.is_member);
    if (!seed) return;
    setBusy(true);
    try {
      const res = await fetch(api(`/confirm/${seed.key}`), { method: "POST" });
      const data: ConfirmResult = await res.json();
      setConfirmed(data);
      const flaggedKeys = new Set(data.flagged.map((f) => f.identity_id));
      setGraph({
        ...graph,
        nodes: graph.nodes.map((n) => ({
          ...n,
          propagated: n.type === "identity" && flaggedKeys.has(n.key),
        })),
      });
    } finally {
      setBusy(false);
    }
  }, [graph]);

  const visibleNodes = graph
    ? view === "network"
      ? graph.nodes
      : // Merchant view: only what a single institution has observed. The
        // evidence is not hidden from the chart, it is absent from the
        // institution that would be scoring.
        graph.nodes.filter((n) => n.institutions.includes(graph.institutions[0]))
    : [];
  const visibleIds = new Set(visibleNodes.map((n) => n.id));
  const visibleEdges = graph
    ? graph.edges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target))
    : [];

  const dormantFlagged = confirmed?.n_dormant_flagged ?? 0;

  return (
    <div className="min-h-screen bg-[#0b0f14] p-6 text-slate-200">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">
            Synthetic Identity as Fraud Infrastructure
          </h1>
          <p className="text-sm text-slate-500">
            Scoring the seam between how an account was born and how it behaves ·{" "}
            <span
              className={status === "connected" ? "text-emerald-400" : "text-amber-400"}
            >
              {status}
            </span>
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/60 p-1">
          {(["merchant", "network"] as ViewScope[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded-md px-4 py-1.5 text-sm capitalize transition ${
                view === v
                  ? "bg-sky-500/20 text-sky-300 ring-1 ring-sky-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {v} view
            </button>
          ))}
        </div>
      </header>

      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Communities" value={String(communities.length)} hint="candidate rings" />
        <Stat
          label="Ring size"
          value={graph ? String(graph.size) : "--"}
          hint={selected ?? "select a ring"}
        />
        <Stat
          label="Flagged by propagation"
          value={confirmed ? String(confirmed.n_flagged) : "--"}
          hint="from one confirmation"
        />
        <Stat
          label="Never transacted"
          value={confirmed ? String(dormantFlagged) : "--"}
          hint="caught before acting"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <aside className="rounded-lg border border-slate-800 bg-slate-900/40">
          <div className="border-b border-slate-800 px-4 py-3 text-sm font-medium text-slate-300">
            Candidate rings
          </div>
          <div className="max-h-[560px] overflow-y-auto">
            {communities.length === 0 && (
              <p className="px-4 py-6 text-sm text-slate-500">
                No communities. Build the graph first:
                <code className="mt-2 block rounded bg-slate-950 px-2 py-1 text-[11px] text-slate-400">
                  POST /admin/build-graph
                </code>
              </p>
            )}
            {communities.map((c) => (
              <button
                key={c.ring_id}
                onClick={() => loadRing(c.ring_id)}
                className={`w-full border-b border-slate-800/60 px-4 py-3 text-left transition hover:bg-slate-800/40 ${
                  selected === c.ring_id ? "bg-sky-500/10" : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-slate-200">{c.ring_id}</span>
                  <span className="text-xs text-slate-500">{c.size} identities</span>
                </div>
                <div className="mt-1 flex items-center gap-3 text-[11px] text-slate-500">
                  <span>score {c.score.toFixed(2)}</span>
                  <span>suspicion {c.suspicion.toFixed(2)}</span>
                  {c.cross_institution && (
                    <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-amber-300">
                      {c.institutions.length} institutions
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </aside>

        <main className="rounded-lg border border-slate-800 bg-slate-900/40">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
            <div className="text-sm text-slate-300">
              {graph ? (
                <>
                  <span className="font-mono">{graph.ring_id}</span>
                  <span className="ml-3 text-slate-500">
                    {visibleNodes.length} nodes · {visibleEdges.length} edges
                    {view === "merchant" && (
                      <span className="ml-2 text-amber-400">
                        ({graph.nodes.length - visibleNodes.length} not visible to one
                        institution)
                      </span>
                    )}
                  </span>
                </>
              ) : (
                <span className="text-slate-500">Select a ring to inspect</span>
              )}
            </div>
            <div className="flex items-center gap-2">
            <button
              onClick={narrateRing}
              disabled={!graph || busy}
              className="rounded-md bg-slate-700/40 px-4 py-1.5 text-sm text-slate-200 ring-1 ring-slate-600 transition hover:bg-slate-700/60 disabled:opacity-40"
            >
              Write the case
            </button>
            <button
              onClick={confirmOne}
              disabled={!graph || busy}
              className="rounded-md bg-rose-500/20 px-4 py-1.5 text-sm text-rose-300 ring-1 ring-rose-500/40 transition hover:bg-rose-500/30 disabled:opacity-40"
            >
              {busy ? "working..." : "Confirm one account as fraud"}
            </button>
            </div>
          </div>

          <div className="h-[520px]">
            {graph ? (
              <ForceGraph2D
                ref={graphRef}
                graphData={{
                  nodes: visibleNodes.map((n) => ({ ...n })),
                  links: visibleEdges.map((e) => ({ ...e })),
                }}
                backgroundColor="#0b0f14"
                nodeRelSize={4}
                nodeVal={(n: any) => (n.type === "identity" ? 3 : 1)}
                nodeColor={(n: any) =>
                  n.propagated ? "#ff4d6d" : NODE_COLOR[n.type] ?? "#64748b"
                }
                nodeLabel={(n: any) =>
                  `${n.type}: ${n.key}${n.propagated ? " — FLAGGED by propagation" : ""}`
                }
                linkColor={() => "rgba(148,163,184,0.18)"}
                linkWidth={(l: any) => Math.min(2, 0.3 + l.weight * 0.1)}
                cooldownTicks={90}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-600">
                No ring selected
              </div>
            )}
          </div>
        </main>
      </div>

      {narrative && (
        <section className="mt-6 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <div className="mb-2 flex flex-wrap items-center gap-3">
            <h2 className="text-sm font-medium text-slate-300">
              Case narrative · <span className="font-mono">{narrative.ring_id}</span>
            </h2>
            <span
              className={`rounded px-2 py-0.5 text-[11px] ${
                narrative.source === "model"
                  ? "bg-sky-500/15 text-sky-300"
                  : "bg-slate-700/40 text-slate-400"
              }`}
            >
              {narrative.source === "model" ? narrative.model : "deterministic"}
            </span>
          </div>
          <div className="space-y-2 whitespace-pre-line text-sm leading-relaxed text-slate-300">
            {narrative.narrative}
          </div>
          <p className="mt-3 text-[11px] text-slate-500">{narrative.note}</p>
        </section>
      )}

      <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_360px]">
        <div className="rounded-lg border border-slate-800 bg-slate-900/40">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-3">
            <div className="text-sm font-medium text-slate-300">
              Live authorisation stream
              <span className="ml-3 text-xs font-normal text-slate-500">
                {items.length ? `${items.length} scored · ` : ""}
                {items.length ? `${(interventionRate * 100).toFixed(1)}% intervened · ` : ""}
                {items.length ? `p99 ${p99Latency.toFixed(1)} ms` : "not started"}
              </span>
            </div>
            <button
              onClick={() => setStreaming((s) => !s)}
              className={`rounded-md px-4 py-1.5 text-sm ring-1 transition ${
                streaming
                  ? "bg-slate-700/40 text-slate-200 ring-slate-600"
                  : "bg-emerald-500/20 text-emerald-300 ring-emerald-500/40 hover:bg-emerald-500/30"
              }`}
            >
              {streaming ? "Stop replay" : "Replay traffic"}
            </button>
          </div>

          <div className="max-h-[320px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-slate-900/95 text-slate-500">
                <tr>
                  <th className="px-3 py-2 font-normal">time</th>
                  <th className="px-3 py-2 font-normal">account</th>
                  <th className="px-3 py-2 font-normal">merchant</th>
                  <th className="px-3 py-2 font-normal">amount</th>
                  <th className="px-3 py-2 font-normal">score</th>
                  <th className="px-3 py-2 font-normal">action</th>
                  <th className="px-3 py-2 font-normal">ms</th>
                </tr>
              </thead>
              <tbody className="font-mono text-slate-300">
                {items.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center font-sans text-slate-500">
                      Press replay to score the scenario as a live stream.
                    </td>
                  </tr>
                )}
                {items.slice(0, 120).map((it) => (
                  <tr
                    key={it.event.event_id}
                    onClick={() => explain(it.event.event_id)}
                    className="cursor-pointer border-t border-slate-800/60 hover:bg-slate-800/40"
                  >
                    <td className="px-3 py-1 text-slate-500">
                      {it.event.ts.slice(11, 19)}
                    </td>
                    <td className="px-3 py-1">{it.event.account_id}</td>
                    <td className="px-3 py-1 text-slate-400">{it.event.merchant_id}</td>
                    <td className="px-3 py-1 tabular-nums text-slate-400">
                      {it.event.amount.toFixed(2)}
                    </td>
                    <td className="px-3 py-1 tabular-nums">{it.decision.score.toFixed(3)}</td>
                    <td className={`px-3 py-1 ${ACTION_COLOR[it.decision.action] ?? ""}`}>
                      {it.decision.action}
                      {it.decision.propagated && (
                        <span className="ml-1 text-[10px] text-rose-400">·ring</span>
                      )}
                    </td>
                    <td className="px-3 py-1 tabular-nums text-slate-500">
                      {it.decision.latency_ms.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {actionCounts.length > 0 && (
            <div className="h-40 border-t border-slate-800 px-2 py-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={actionCounts}>
                  <CartesianGrid stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="action" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} width={40} />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="count" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-900/40">
          <div className="border-b border-slate-800 px-4 py-3 text-sm font-medium text-slate-300">
            Explain this decision
          </div>
          {explanation ? (
            <div className="space-y-3 px-4 py-3 text-sm">
              <div className="font-mono text-xs text-slate-500">{explanation.event_id}</div>
              <div className="flex items-center gap-3">
                <span className="text-2xl font-semibold tabular-nums text-slate-100">
                  {explanation.score.toFixed(3)}
                </span>
                <span className={ACTION_COLOR[explanation.action] ?? "text-slate-400"}>
                  {explanation.action}
                </span>
                <span className="text-xs text-slate-500">band {explanation.band}</span>
              </div>
              <ul className="space-y-1.5 text-xs text-slate-400">
                {explanation.reasons.length === 0 && (
                  <li className="text-slate-600">
                    Nothing fired: the score sat below every rule threshold.
                  </li>
                )}
                {explanation.reasons.map((r) => (
                  <li key={r} className="border-l-2 border-slate-700 pl-2">
                    {r}
                  </li>
                ))}
              </ul>
              {explanation.evidence_path.length > 0 && (
                <div>
                  <div className="mb-1 text-[11px] uppercase tracking-wider text-slate-500">
                    Evidence path
                  </div>
                  <ol className="space-y-1 font-mono text-[11px] text-slate-400">
                    {explanation.evidence_path.map((h, i) => (
                      <li key={i}>
                        {h.via} → {h.to_node} ({h.contribution.toFixed(3)})
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ) : (
            <p className="px-4 py-6 text-sm text-slate-500">
              Select an event from the stream. Every decision carries its reason codes,
              because a block a human cannot explain is a block a regulator will not accept.
            </p>
          )}
        </div>
      </section>

      {confirmed && (
        <section className="mt-6 rounded-lg border border-rose-900/50 bg-rose-950/20 p-4">
          <h2 className="text-sm font-medium text-rose-300">
            Retro-propagation from {confirmed.seed}
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            One confirmation raised <strong className="text-slate-200">{confirmed.n_flagged}</strong>{" "}
            sibling accounts, of which{" "}
            <strong className="text-rose-300">{dormantFlagged}</strong> have never transacted —
            no behavioural signal exists for those, so nothing else could reach them.
          </p>
          <div className="mt-3 max-h-56 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 font-normal">identity</th>
                  <th className="py-1 font-normal">score</th>
                  <th className="py-1 font-normal">hops</th>
                  <th className="py-1 font-normal">status</th>
                  <th className="py-1 font-normal">evidence</th>
                </tr>
              </thead>
              <tbody className="font-mono text-slate-300">
                {confirmed.flagged.slice(0, 40).map((f) => (
                  <tr key={f.identity_id} className="border-t border-slate-800/60">
                    <td className="py-1">{f.identity_id}</td>
                    <td className="py-1 tabular-nums">{f.score.toFixed(3)}</td>
                    <td className="py-1 tabular-nums">{f.hops}</td>
                    <td className="py-1">
                      {f.dormant ? (
                        <span className="text-rose-300">never transacted</span>
                      ) : (
                        <span className="text-slate-500">active</span>
                      )}
                    </td>
                    <td className="py-1 text-slate-500">
                      {f.path.map((h) => h.via).join(" → ") || "direct"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

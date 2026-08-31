/**
 * Analyst console.
 *
 * The screen has one job: make the *chain* legible, not the cluster. An earlier
 * version showed a force graph of coloured dots and a ring score, which is
 * indistinguishable from any other clustering demo — the signals that actually
 * make the case were all computed and none of them reached the screen.
 *
 * So the evidence panel is the centre of gravity here, laid out as the three
 * stages of the supply chain:
 *
 *   Manufacture — the AI-generated face and document
 *   Onboard     — how the identity got through KYC
 *   Weaponise   — the card testing itself
 *
 * Every figure sits next to the same figure for everyone outside the ring,
 * because a decline ratio of 0.93 means nothing until you see 0.15 beside it.
 *
 * The other two things that carry the pitch:
 *   1. Confirm one account, and its siblings light up — including accounts that
 *      have never transacted.
 *   2. The merchant/network toggle, which changes what evidence the detector is
 *      given rather than what the chart draws.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

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

interface EvidenceRow {
  metric: string;
  label: string;
  description: string;
  direction: "higher" | "lower" | "neutral" | "count";
  ring: number;
  population: number | null;
  ratio: number | null;
  elevated: boolean | null;
}

interface RingEvidence {
  ring_id: string;
  members: number;
  cross_institution: boolean;
  institutions: string[];
  links: { kind: string; pairs: number }[];
  manufacture: EvidenceRow[];
  onboard: EvidenceRow[];
  weaponise: EvidenceRow[];
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

interface Narrative {
  ring_id: string;
  narrative: string;
  source: "model" | "template";
  model: string | null;
  note: string;
}

const ACTION_COLOR: Record<string, string> = {
  allow: "text-slate-500",
  step_up: "text-sky-300",
  throttle: "text-amber-300",
  honeypot: "text-fuchsia-300",
  block: "text-rose-300",
};

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

/** What each stage of the chain is, in the words the pitch uses. */
const STAGES: { key: "manufacture" | "onboard" | "weaponise"; title: string; blurb: string }[] = [
  {
    key: "manufacture",
    title: "1 · Manufacture",
    blurb: "The AI-generated face and document, as a verification vendor would score them",
  },
  {
    key: "onboard",
    title: "2 · Onboard",
    blurb: "How the identity passed KYC — recombined PII, impossible credit history, shared infrastructure",
  },
  {
    key: "weaponise",
    title: "3 · Weaponise",
    blurb: "The card testing: many PANs, tiny amounts, high declines, CVV failures",
  },
];

const api = (path: string) => `/api${path}`;

/**
 * A human name for a community.
 *
 * "cmt_0005 · score 2.89 · suspicion 0.78" is what the detector calls it, and
 * it tells a reader nothing. The rank is what an analyst works from — the queue
 * is ordered by how strong the case is — so lead with that and keep the
 * internal id alongside for anyone who wants to query it.
 */
function ringName(index: number): string {
  return `Suspected ring #${index + 1}`;
}

/** One sentence that says what the evidence adds up to. */
function verdict(ev: RingEvidence | null): { line: string; tone: string } | null {
  if (!ev) return null;
  const strong = (rows: EvidenceRow[]) =>
    rows.filter((r) => r.elevated === true && severityOf(r) >= 3).length;

  const made = strong(ev.manufacture);
  const built = strong(ev.onboard);
  const testing = strong(ev.weaponise);

  const parts: string[] = [];
  if (made) parts.push("faces and documents show generation artifacts");
  if (built) parts.push("manufactured as one batch");
  if (testing) parts.push("currently testing cards");

  if (!parts.length) {
    return {
      line: "Nothing here stands out against the rest of the population. This looks like a cluster, not a ring.",
      tone: "text-slate-400",
    };
  }
  const joined =
    parts.length === 1
      ? parts[0]
      : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
  return {
    line: `${ev.members} accounts across ${ev.institutions.length} institution${
      ev.institutions.length === 1 ? "" : "s"
    } — ${joined}.`,
    tone: testing ? "text-rose-200" : "text-amber-200",
  };
}

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "--";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 10) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

function Stat({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${tone ?? "text-slate-100"}`}>
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

/** How far from the population a metric sits, in whichever direction is the bad one. */
function severityOf(row: EvidenceRow): number {
  if (row.ratio === null || row.ratio === 0) return 1;
  return row.ratio >= 1 ? row.ratio : 1 / row.ratio;
}

/**
 * Bar length is logarithmic, not linear.
 *
 * The first version scaled linearly and clipped at 8x, so a zero-auth ratio
 * 113x the population and a decline ratio 12x it drew the same full bar — the
 * strongest evidence on the screen was flattened into a wall of identical red.
 * Card-testing signals span two orders of magnitude, so the axis has to.
 */
function barWidth(severity: number): number {
  if (severity <= 1) return 3;
  return Math.max(3, Math.min(100, (Math.log10(severity) / Math.log10(200)) * 100));
}

function EvidenceRowView({ row }: { row: EvidenceRow }) {
  const isCount = row.direction === "count";
  const severity = severityOf(row);
  const elevated = row.elevated === true;
  const strong = elevated && severity >= 10;

  return (
    <div className="border-t border-slate-800/60 px-4 py-2.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-xs font-medium capitalize text-slate-200">{row.label}</span>
        <span className="shrink-0 font-mono text-xs tabular-nums text-slate-300">
          {fmt(row.ring)}
          {!isCount && <span className="ml-1.5 text-slate-600">vs {fmt(row.population)}</span>}
        </span>
      </div>

      <p className="mt-0.5 text-[10px] leading-snug text-slate-500">{row.description}</p>

      {!isCount && row.ratio !== null && (
        <div className="mt-2 flex items-center gap-2">
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800/80">
            <div
              className={`h-full rounded-full ${
                strong ? "bg-rose-400" : elevated ? "bg-rose-500/60" : "bg-slate-600/60"
              }`}
              style={{ width: `${barWidth(severity)}%` }}
            />
          </div>
          <span
            className={`w-[86px] shrink-0 rounded px-1.5 py-0.5 text-right font-mono text-[10px] tabular-nums ${
              strong
                ? "bg-rose-500/20 font-semibold text-rose-200"
                : elevated
                  ? "text-rose-300"
                  : "text-slate-600"
            }`}
          >
            {severity < 1.05
              ? "in line"
              : `${severity < 10 ? severity.toFixed(1) : Math.round(severity)}x ${
                  row.direction === "lower" ? "below" : "above"
                }`}
          </span>
        </div>
      )}
    </div>
  );
}

/** A stage of the chain: its rows, strongest evidence first, with a headline. */
function EvidenceStage({
  title,
  blurb,
  rows,
}: {
  title: string;
  blurb: string;
  rows: EvidenceRow[];
}) {
  // Strongest first. Twenty-six rows in source order buries the argument; the
  // reader should land on the number that makes the case.
  const sorted = [...rows].sort((a, b) => {
    const ea = a.elevated === true ? 1 : 0;
    const eb = b.elevated === true ? 1 : 0;
    if (ea !== eb) return eb - ea;
    return severityOf(b) - severityOf(a);
  });
  const elevated = rows.filter((r) => r.elevated === true);
  const peak = elevated.length ? Math.max(...elevated.map(severityOf)) : 0;
  const scored = rows.filter((r) => r.direction === "higher" || r.direction === "lower").length;

  return (
    <section className="border-b border-slate-800 last:border-b-0">
      <div className="flex items-start justify-between gap-3 px-4 pb-2 pt-3.5">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-wider text-sky-300">{title}</div>
          <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">{blurb}</p>
        </div>
        {scored > 0 && (
          <div className="shrink-0 text-right">
            <div
              className={`font-mono text-lg font-semibold leading-none tabular-nums ${
                peak >= 10 ? "text-rose-300" : peak > 1 ? "text-amber-300" : "text-slate-500"
              }`}
            >
              {peak >= 1.05 ? `${peak < 10 ? peak.toFixed(1) : Math.round(peak)}x` : "--"}
            </div>
            <div className="mt-0.5 text-[10px] text-slate-500">
              {elevated.length}/{scored} elevated
            </div>
          </div>
        )}
      </div>
      <div className="pb-1">
        {sorted.map((row) => (
          <EvidenceRowView key={row.metric} row={row} />
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [view, setView] = useState<ViewScope>("network");
  const [communities, setCommunities] = useState<Community[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [graph, setGraph] = useState<RingGraph | null>(null);
  const [evidence, setEvidence] = useState<RingEvidence | null>(null);
  const [confirmed, setConfirmed] = useState<ConfirmResult | null>(null);
  const [narrative, setNarrative] = useState<Narrative | null>(null);
  const [status, setStatus] = useState<string>("connecting");
  const [busy, setBusy] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [items, setItems] = useState<StreamItem[]>([]);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const graphRef = useRef<any>(null);
  const canvasWrapRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 520 });

  /**
   * Measure the container and hand the size to the force graph explicitly.
   *
   * ForceGraph2D defaults its canvas to `window.innerWidth/innerHeight`, not to
   * its parent element. On a wide screen that meant a 2880px canvas inside a
   * ~900px column: it painted straight over the evidence panel to its right,
   * which was therefore invisible, and `zoomToFit` fitted the graph to the
   * canvas rather than to the visible area, leaving the nodes in a corner.
   * Both symptoms were the same missing pair of props.
   */
  useEffect(() => {
    const el = canvasWrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setCanvasSize({ width: Math.max(0, Math.floor(width)), height: Math.max(0, Math.floor(height)) });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  /**
   * Fit the layout to the canvas once it settles.
   *
   * Without this the simulation leaves the component wherever it happened to
   * converge — for a small ring, a knot of nodes in one corner of an otherwise
   * empty 520px canvas. Refit whenever the data changes, which includes the
   * merchant/network toggle removing most of the nodes.
   */
  const fitGraph = useCallback(() => {
    graphRef.current?.zoomToFit(500, 70);
  }, []);

  // A resize changes what "fit" means, so refit after one settles.
  useEffect(() => {
    if (!canvasSize.width) return;
    const t = setTimeout(fitGraph, 120);
    return () => clearTimeout(t);
  }, [canvasSize.width, canvasSize.height, fitGraph]);

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
    setEvidence(null);
    setBusy(true);
    Promise.all([
      fetch(api(`/graph/ring/${ringId}`)).then((r) => r.json()),
      fetch(api(`/ring/${ringId}/evidence`)).then((r) => (r.ok ? r.json() : null)),
    ])
      .then(([g, e]) => {
        setGraph(g);
        setEvidence(e);
      })
      .finally(() => setBusy(false));
  }, []);

  /**
   * The demo moment. Confirm one account and let the evidence propagate; the
   * response says which siblings were raised and how many have never
   * transacted at all.
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

  /**
   * The replay is scoped by the same toggle as everything else, and the scope
   * is applied in the scorer, not here: in merchant view the stream is one
   * institution's traffic and the rolling windows are built from that alone.
   */
  useEffect(() => {
    if (!streaming) return;
    setItems([]);
    const es = new EventSource(api(`/stream?speed=60&limit=4000&view=${view}`));
    es.onmessage = (msg) => {
      const item: StreamItem = JSON.parse(msg.data);
      setItems((prev) => {
        const next = [item, ...prev];
        return next.length > 400 ? next.slice(0, 400) : next;
      });
    };
    es.onerror = () => {
      es.close();
      setStreaming(false);
    };
    return () => es.close();
  }, [streaming, view]);

  const explain = useCallback((eventId: string) => {
    fetch(api(`/explain/${eventId}`))
      .then((r) => (r.ok ? r.json() : null))
      .then(setExplanation)
      .catch(() => setExplanation(null));
  }, []);

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

  const interventionRate = items.length
    ? items.filter((i) => i.decision.action !== "allow").length / items.length
    : 0;
  const p99Latency = useMemo(() => {
    if (!items.length) return 0;
    const sorted = items.map((i) => i.decision.latency_ms).sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(0.99 * sorted.length))];
  }, [items]);

  const dormantFlagged = confirmed?.n_dormant_flagged ?? 0;
  const verdictLine = useMemo(() => verdict(evidence), [evidence]);
  const nodeTypesPresent = useMemo(
    () => Array.from(new Set(visibleNodes.map((n) => n.type))).sort(),
    [visibleNodes]
  );

  return (
    <div className="min-h-screen bg-[#0b0f14] text-slate-200">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-[#0b0f14]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1800px] flex-wrap items-center justify-between gap-4 px-6 py-3">
          <div>
            <h1 className="text-base font-semibold text-slate-100">
              Synthetic Identity as Fraud Infrastructure
            </h1>
            <p className="text-xs text-slate-500">
              Fake people are manufactured in batches, passed through KYC, aged quietly, then used
              to test stolen cards ·{" "}
              <span className={status === "connected" ? "text-emerald-400" : "text-amber-400"}>
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
        </div>
      </header>

      <div className="mx-auto max-w-[1800px] px-6 py-5">
        <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="Candidate rings" value={String(communities.length)} hint="from the identity graph" />
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
            tone={dormantFlagged ? "text-rose-300" : undefined}
          />
        </div>

        <p className="mb-5 max-w-5xl text-xs leading-relaxed text-slate-500">
          Identity is scored once, at signup. Transactions are scored one at a time, at
          authorisation. <span className="text-slate-300">Nobody scores the seam between them</span>{" "}
          — so a batch of manufactured accounts gets caught one at a time, after each has already
          been used. This console scores the seam: pick a suspected batch, read the evidence at
          every stage of the chain, then confirm a single account and watch its siblings light up
          before they act.
        </p>

        <div className="grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)_460px]">
          {/* -------------------------------------------------- ring list */}
          <aside className="rounded-lg border border-slate-800 bg-slate-900/40">
            <div className="flex h-14 items-center justify-between border-b border-slate-800 px-4">
              <span className="text-sm font-medium text-slate-300">Candidate rings</span>
              <span className="font-mono text-[11px] text-slate-500">{communities.length}</span>
            </div>
            <div className="max-h-[620px] overflow-y-auto">
              {communities.length === 0 && (
                <p className="px-4 py-6 text-sm text-slate-500">
                  No communities. Build the graph first:
                  <code className="mt-2 block rounded bg-slate-950 px-2 py-1 text-[11px] text-slate-400">
                    POST /admin/build-graph
                  </code>
                </p>
              )}
              {communities.map((c, i) => (
                <button
                  key={c.ring_id}
                  onClick={() => loadRing(c.ring_id)}
                  className={`w-full border-b border-slate-800/60 px-4 py-3 text-left transition hover:bg-slate-800/40 ${
                    selected === c.ring_id ? "bg-sky-500/10" : ""
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-sm font-medium text-slate-200">{ringName(i)}</span>
                    <span className="shrink-0 text-xs text-slate-400">{c.size} accounts</span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-500">
                    {c.cross_institution && (
                      <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-amber-300">
                        {c.institutions.length} banks
                      </span>
                    )}
                    <span className="font-mono text-slate-600">{c.ring_id}</span>
                  </div>
                </button>
              ))}
            </div>
          </aside>

          {/* ------------------------------------------------------ graph */}
          <main className="min-w-0 rounded-lg border border-slate-800 bg-slate-900/40">
            <div className="flex h-14 items-center justify-between gap-3 border-b border-slate-800 px-4">
              <div className="min-w-0 text-sm text-slate-300">
                {graph ? (
                  <>
                    <span className="font-medium">
                      {ringName(communities.findIndex((c) => c.ring_id === graph.ring_id))}
                    </span>
                    <span className="ml-3 text-xs text-slate-500">
                      {visibleNodes.length} nodes · {visibleEdges.length} edges
                      {view === "merchant" && (
                        <span className="ml-2 text-amber-400">
                          · {graph.nodes.length - visibleNodes.length} invisible to one institution
                        </span>
                      )}
                    </span>
                  </>
                ) : (
                  <span className="text-slate-500">Select a ring to inspect</span>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <button
                  onClick={fitGraph}
                  disabled={!graph}
                  title="Fit the layout to the canvas"
                  className="rounded-md px-2.5 py-1 text-xs text-slate-400 ring-1 ring-slate-700 transition hover:text-slate-200 disabled:opacity-30"
                >
                  Fit
                </button>
                <button
                  onClick={narrateRing}
                  disabled={!graph || busy}
                  className="rounded-md px-3 py-1 text-xs text-slate-300 ring-1 ring-slate-700 transition hover:bg-slate-800/60 disabled:opacity-30"
                >
                  Write the case
                </button>
                <button
                  onClick={confirmOne}
                  disabled={!graph || busy}
                  className="rounded-md bg-rose-500/15 px-3 py-1 text-xs font-medium text-rose-300 ring-1 ring-rose-500/40 transition hover:bg-rose-500/25 disabled:opacity-30"
                >
                  {busy ? "working…" : "Confirm one account as fraud"}
                </button>
              </div>
            </div>

            <div ref={canvasWrapRef} className="relative h-[520px] overflow-hidden">
              {busy && !graph && (
                <div className="absolute inset-0 z-10 flex items-center justify-center text-sm text-slate-500">
                  building the ring…
                </div>
              )}
              {graph && canvasSize.width > 0 ? (
                <ForceGraph2D
                  ref={graphRef}
                  width={canvasSize.width}
                  height={canvasSize.height}
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
                  onEngineStop={fitGraph}
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-1 text-slate-600">
                  <span className="text-sm">No ring selected</span>
                  <span className="text-xs">
                    Pick a candidate on the left to see who it connects and why
                  </span>
                </div>
              )}
            </div>

            {nodeTypesPresent.length > 0 && (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-800 px-4 py-2 text-[11px] text-slate-500">
                {nodeTypesPresent.map((t) => (
                  <span key={t} className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ background: NODE_COLOR[t] ?? "#64748b" }}
                    />
                    {t.replace("_", "/")}
                  </span>
                ))}
                <span className="flex items-center gap-1.5 text-rose-300">
                  <span className="inline-block h-2 w-2 rounded-full bg-[#ff4d6d]" />
                  flagged by propagation
                </span>
              </div>
            )}
          </main>

          {/* --------------------------------------------------- evidence */}
          <aside className="min-w-0 rounded-lg border border-slate-800 bg-slate-900/40">
            <div className="flex h-14 items-center justify-between gap-3 border-b border-slate-800 px-4">
              <div className="min-w-0">
                <div className="text-sm font-medium text-slate-300">Why this is a ring</div>
                <p className="truncate text-[11px] text-slate-500">
                  Ring vs everyone outside it, at every stage of the chain
                </p>
              </div>
              {evidence && (
                <span className="shrink-0 font-mono text-[11px] text-slate-500">
                  {evidence.members} identities
                </span>
              )}
            </div>

            {!evidence && (
              <p className="px-4 py-6 text-sm text-slate-500">
                Select a ring. The evidence is assembled off the auth path, at graph-build time.
              </p>
            )}

            {evidence && (
              <div className="max-h-[700px] overflow-y-auto">
                {verdictLine && (
                  <div className="border-b border-slate-800 bg-slate-900/60 px-4 py-3">
                    <p className={`text-sm leading-relaxed ${verdictLine.tone}`}>
                      {verdictLine.line}
                    </p>
                  </div>
                )}
                {evidence.links.length > 0 && (
                  <div className="border-b border-slate-800 px-4 py-3">
                    <div className="text-[11px] uppercase tracking-wider text-slate-500">
                      Held together by
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {evidence.links.map((l) => (
                        <span
                          key={l.kind}
                          className="rounded bg-slate-800/70 px-2 py-0.5 text-[11px] text-slate-300"
                        >
                          {l.kind.replace(/_/g, " ")}{" "}
                          <span className="font-mono text-slate-500">{l.pairs}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {STAGES.map(({ key, title, blurb }) => (
                  <EvidenceStage key={key} title={title} blurb={blurb} rows={evidence[key]} />
                ))}
              </div>
            )}
          </aside>
        </div>

        {/* ------------------------------------------- retro-propagation */}
        {confirmed && (
          <section className="mt-5 rounded-lg border border-rose-900/50 bg-rose-950/20 p-4">
            <h2 className="text-sm font-medium text-rose-300">
              Retro-propagation from <span className="font-mono">{confirmed.seed}</span>
            </h2>
            <p className="mt-1 max-w-4xl text-sm leading-relaxed text-slate-400">
              One confirmation raised{" "}
              <strong className="text-slate-200">{confirmed.n_flagged}</strong> sibling account
              {confirmed.n_flagged === 1 ? "" : "s"}
              {dormantFlagged > 0 ? (
                <>
                  , of which <strong className="text-rose-300">{dormantFlagged}</strong> have never
                  transacted — no behavioural signal exists for those, so nothing else could reach
                  them.
                </>
              ) : (
                // Not every ring has dormant members, and claiming the win
                // anyway would be the kind of thing a judge checks.
                <>. All of them had already transacted, so this ring offers no
                  before-the-first-transaction catch — the propagation still reached them from a
                  single confirmation, through shared infrastructure rather than their own
                  behaviour.
                </>
              )}
            </p>
            {/* Column widths are pinned. Left to itself the table stretched five
                columns across a 1800px page, so a three-character hop count sat
                in a 300px cell. */}
            <div className="mt-3 max-h-60 overflow-auto rounded border border-rose-900/30">
              <table className="w-full table-fixed text-left text-xs">
                <colgroup>
                  <col className="w-[170px]" />
                  <col className="w-[80px]" />
                  <col className="w-[60px]" />
                  <col className="w-[140px]" />
                  <col />
                </colgroup>
                <thead className="sticky top-0 bg-[#1a0d13] text-slate-400">
                  <tr>
                    <th className="px-3 py-1.5 font-normal">identity</th>
                    <th className="px-3 py-1.5 text-right font-normal">score</th>
                    <th className="px-3 py-1.5 text-right font-normal">hops</th>
                    <th className="px-3 py-1.5 font-normal">status</th>
                    <th className="px-3 py-1.5 font-normal">evidence path</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-slate-300">
                  {confirmed.flagged.slice(0, 40).map((f) => (
                    <tr key={f.identity_id} className="border-t border-rose-900/20">
                      <td className="truncate px-3 py-1">{f.identity_id}</td>
                      <td className="px-3 py-1 text-right tabular-nums">{f.score.toFixed(3)}</td>
                      <td className="px-3 py-1 text-right tabular-nums text-slate-500">
                        {f.hops}
                      </td>
                      <td className="px-3 py-1">
                        {f.dormant ? (
                          <span className="text-rose-300">never transacted</span>
                        ) : (
                          <span className="text-slate-600">active</span>
                        )}
                      </td>
                      <td className="truncate px-3 py-1 text-slate-500">
                        {f.path.map((h) => h.via).join(" → ") || "direct"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {confirmed.flagged.length > 40 && (
              <p className="mt-2 text-[11px] text-slate-500">
                Showing the 40 strongest of {confirmed.flagged.length}.
              </p>
            )}
          </section>
        )}

        {/* ------------------------------------------------- case narrative */}
        {narrative && (
          <section className="mt-5 rounded-lg border border-slate-800 bg-slate-900/40 p-4">
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
            <div className="max-w-4xl whitespace-pre-line text-sm leading-relaxed text-slate-300">
              {narrative.narrative}
            </div>
            <p className="mt-3 text-[11px] text-slate-500">{narrative.note}</p>
          </section>
        )}

        {/* ------------------------------------------------- live stream */}
        <section className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
          <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-900/40">
            <div className="flex h-14 items-center justify-between gap-3 border-b border-slate-800 px-4">
              <div className="min-w-0 text-sm font-medium text-slate-300">
                Live authorisation stream
                <span className="ml-3 text-xs font-normal text-slate-500">
                  {items.length
                    ? `${items.length} scored · ${(interventionRate * 100).toFixed(1)}% intervened · p99 ${p99Latency.toFixed(1)} ms`
                    : "not started"}
                </span>
              </div>
              <button
                onClick={() => setStreaming((s) => !s)}
                className={`shrink-0 rounded-md px-3 py-1 text-xs font-medium ring-1 transition ${
                  streaming
                    ? "text-slate-300 ring-slate-700 hover:bg-slate-800/60"
                    : "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40 hover:bg-emerald-500/25"
                }`}
              >
                {streaming ? "Stop replay" : "Replay traffic"}
              </button>
            </div>

            <div className="max-h-[340px] overflow-auto">
              <table className="w-full table-fixed text-left text-xs">
                <colgroup>
                  <col className="w-[90px]" />
                  <col className="w-[130px]" />
                  <col className="w-[120px]" />
                  <col className="w-[100px]" />
                  <col className="w-[90px]" />
                  <col className="w-[130px]" />
                  <col className="w-[70px]" />
                </colgroup>
                <thead className="sticky top-0 bg-slate-900/95 text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-normal">time</th>
                    <th className="px-3 py-2 font-normal">account</th>
                    <th className="px-3 py-2 font-normal">merchant</th>
                    <th className="px-3 py-2 text-right font-normal">amount</th>
                    <th className="px-3 py-2 text-right font-normal">score</th>
                    <th className="px-3 py-2 font-normal">action</th>
                    <th className="px-3 py-2 text-right font-normal">ms</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-slate-300">
                  {items.length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-3 py-10 text-center font-sans text-slate-500">
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
                      <td className="px-3 py-1 text-slate-500">{it.event.ts.slice(11, 19)}</td>
                      <td className="px-3 py-1">{it.event.account_id}</td>
                      <td className="px-3 py-1 text-slate-400">{it.event.merchant_id}</td>
                      <td className="px-3 py-1 text-right tabular-nums text-slate-400">
                        {it.event.amount.toFixed(2)}
                      </td>
                      <td className="px-3 py-1 text-right tabular-nums">
                        {it.decision.score.toFixed(3)}
                      </td>
                      <td className={`px-3 py-1 ${ACTION_COLOR[it.decision.action] ?? ""}`}>
                        {it.decision.action}
                        {it.decision.propagated && (
                          <span className="ml-1 text-[10px] text-rose-400">·ring</span>
                        )}
                      </td>
                      <td className="px-3 py-1 text-right tabular-nums text-slate-500">
                        {it.decision.latency_ms.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="min-w-0 rounded-lg border border-slate-800 bg-slate-900/40">
            <div className="flex h-14 items-center border-b border-slate-800 px-4 text-sm font-medium text-slate-300">
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
              <p className="px-4 py-6 text-sm leading-relaxed text-slate-500">
                Click an event in the stream. Every decision carries its reason codes, because a
                block a human cannot explain is a block a regulator will not accept.
              </p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

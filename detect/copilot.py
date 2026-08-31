"""Analyst copilot: a case narrative, and reason codes in plain English.

Two jobs, both **off the authorisation path**. The design doc is explicit that
an LLM must never sit in the auth round trip -- the latency budget is ~50 ms for
scoring inside a few hundred for the whole authorisation, and a model call is
two orders of magnitude away from that. Saying so out loud is part of the pitch.
What an LLM is genuinely good at here is the thing that happens *after* a case
lands in a queue: reading a subgraph, a feature vector and a propagation path,
and writing the paragraph an analyst would otherwise write themselves.

The rule this module enforces is that **the model never sees PII and never
decides anything**. It receives tokens, tags, counts and scores -- the same
things that cross the L1 boundary -- and it produces prose. Every number in the
prose is passed in; nothing is computed by the model. If it hallucinates a
figure, the evidence block sitting beside it in the console contradicts it
visibly, which is the correct failure mode for an assistive tool.

Without an API key the deterministic template runs instead, so the console has
no dependency on a network call and the demo cannot fail on a missing secret.
The template is genuinely usable -- it is what the endpoint returns by default.

Usage::

    python -m detect.copilot --data data/sloppy --ring ring_000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Read ``.env`` from the repo root into the environment, if it exists.

    Hand-rolled rather than adding ``python-dotenv``: the whole job is fifteen
    lines, and the project otherwise has no dependency that a plain
    ``pip install lightgbm fastapi`` does not already bring in. Existing
    environment variables win, so an exported key still overrides the file.
    """
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

#: Haiku is the right size for this job. The evidence is already assembled and
#: already quantified -- the model is writing two paragraphs over a JSON block,
#: not reasoning about fraud -- and it is roughly a fifth the cost of Opus.
#: Override with ``FRAUD_NARRATIVE_MODEL`` to run the narrative on a larger
#: model for a pitch.
DEFAULT_MODEL = "claude-haiku-4-5"
MODEL = os.environ.get("FRAUD_NARRATIVE_MODEL", DEFAULT_MODEL)

#: Models that accept ``output_config.effort`` and server-side refusal
#: fallbacks. Haiku 4.5 rejects both with a 400, so the request shape has to
#: follow the model rather than being fixed.
_FRONTIER_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)

#: Kept small on purpose. A case narrative is two paragraphs; a large ceiling
#: only buys the chance of an essay in a queue that an analyst reads at speed.
MAX_TOKENS = 1500

SYSTEM = """\
You write case narratives for fraud analysts at a payment network. Your reader \
is an expert who will act on what you write, so be precise and unhedged, and \
never pad.

Rules that matter more than style:

* Use ONLY the evidence given to you. Every figure in your narrative must \
appear in that evidence. Do not estimate, extrapolate, or supply a number that \
is missing -- say it is not available.
* You are not deciding anything. The decision has been made by the scoring \
system; you are explaining what the evidence shows and what it does not.
* Identifiers are tokens and hashes, not people. There are no names, addresses \
or card numbers in your input, and you must not invent any.
* Say plainly where the evidence is weak. An analyst who is misled by a \
confident narrative loses more time than one given an honest uncertainty.

Write two short paragraphs and then a line beginning "Recommended next step:". \
The first paragraph is what happened; the second is what links the accounts and \
how strongly. No headings, no bullet lists, no preamble."""


@dataclass
class CaseEvidence:
    """Everything the narrative may draw on. No PII crosses this boundary."""

    ring_id: str
    ring_size: int
    cohesion: float
    suspicion: float
    institutions: list[str]
    cross_institution: bool
    #: Shared-attribute counts that made this a community, e.g. {"device": 12}.
    shared_attributes: dict[str, int] = field(default_factory=dict)
    #: The confirmed account the evidence propagated from, if any.
    seed_identity: str | None = None
    n_flagged: int = 0
    n_dormant_flagged: int = 0
    #: Representative propagation paths: [{"identity", "hops", "via", "score"}].
    evidence_paths: list[dict] = field(default_factory=list)
    #: Behavioural features of the ring's testing traffic, already aggregated.
    behaviour: dict[str, float] = field(default_factory=dict)
    #: The three supply-chain stages from ``detect.evidence``, each a list of
    #: {metric, ring, population, ratio, elevated}. Without these the narrative
    #: had the graph and nothing else -- the model's own first draft said "no
    #: transactional behaviour data is available", which was true and was the
    #: reason to pass them.
    stages: dict[str, list] = field(default_factory=dict)
    #: Reason codes that fired, as raw codes; text is looked up separately.
    reason_codes: list[str] = field(default_factory=list)
    decision: str | None = None
    score: float | None = None


def _reason_lines(codes: list[str]) -> list[str]:
    from contracts.decisions import REASON_TEXT, ReasonCode

    out = []
    for c in codes:
        try:
            out.append(f"{c}: {REASON_TEXT[ReasonCode(c)]}")
        except (KeyError, ValueError):
            out.append(c)
    return out


def _plural(word: str, n: int) -> str:
    """Enough English for the node types we actually have: address -> addresses."""
    if n == 1:
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def template_narrative(ev: CaseEvidence) -> str:
    """The deterministic narrative. No model, no key, no network.

    This is the default the console renders, so it has to stand on its own
    rather than read as a placeholder waiting for the real thing.
    """
    shared = ", ".join(
        f"{n} shared {_plural(k.replace('_', ' '), n)}"
        for k, n in sorted(ev.shared_attributes.items(), key=lambda kv: -kv[1])
    )
    inst = (
        f"across {len(ev.institutions)} institutions ({', '.join(ev.institutions)})"
        if ev.cross_institution
        else f"within {ev.institutions[0] if ev.institutions else 'one institution'}"
    )

    first = (
        f"Community {ev.ring_id} groups {ev.ring_size} identities {inst}, with "
        f"cohesion {ev.cohesion:.2f} and shared-attribute suspicion {ev.suspicion:.2f}."
    )
    if shared:
        first += f" The community is held together by {shared}."
    if ev.behaviour:
        bits = ", ".join(f"{k.replace('_', ' ')} {v:.3g}" for k, v in sorted(ev.behaviour.items()))
        first += f" Its authorisation traffic shows {bits}."

    stage_lines = []
    for stage, phrase in (
        ("manufacture", "Their faces and documents run"),
        ("onboard", "Their applications run"),
        # Singular subject, so it does not read "their traffic run".
        ("weaponise", "Their authorisation traffic runs"),
    ):
        top = _top_signals(ev.stages.get(stage, []))
        if top:
            stage_lines.append(f"{phrase} {top}.")
    if stage_lines:
        first += " " + " ".join(stage_lines)

    if ev.seed_identity:
        second = (
            f"Confirmation of {ev.seed_identity} propagated to {ev.n_flagged} sibling "
            f"identities, of which {ev.n_dormant_flagged} have never transacted at all. "
            "Those have no behavioural signal of any kind, so no transaction-level "
            "model could have reached them; they are reachable only through shared "
            "infrastructure."
        )
        if ev.evidence_paths:
            p = ev.evidence_paths[0]
            second += (
                f" The strongest path runs {p.get('hops', '?')} hop(s) via "
                f"{p.get('via', 'shared attributes')} at contribution "
                f"{float(p.get('score', 0.0)):.3f}."
            )
    else:
        second = (
            "No account in this community has been confirmed, so nothing has been "
            "propagated. The community is a candidate on structure alone, and "
            "structure alone is not evidence of fraud -- a family, a shared office "
            "or a reused public IP produces the same shape."
        )

    reasons = _reason_lines(ev.reason_codes)
    if reasons:
        second += " Reason codes on the decision: " + "; ".join(reasons)

    if ev.n_dormant_flagged:
        step = (
            f"Recommended next step: review the {ev.n_dormant_flagged} dormant "
            "identities before they transact, starting with the shortest evidence paths."
        )
    elif ev.seed_identity:
        step = (
            "Recommended next step: confirm or clear the highest-scoring siblings, "
            "since each verdict re-weights the rest of the community."
        )
    else:
        step = (
            "Recommended next step: hold for a confirmed account. Acting on structure "
            "with no confirmation is how legitimate clusters get frozen."
        )

    return f"{first}\n\n{second}\n\n{step}"


def _top_signals(rows: list, limit: int = 3) -> str:
    """The strongest few signals in a stage, as a readable clause."""
    scored = []
    for r in rows:
        ratio = r.get("ratio")
        if not r.get("elevated") or not ratio:
            continue
        severity = ratio if ratio >= 1 else 1 / ratio
        # "1x above the population" is not a finding. Only quote a gap that a
        # reader would act on, and keep a decimal below 10 so 1.4x and 9.8x do
        # not both round to the same claim.
        if severity < 1.5:
            continue
        shown = f"{severity:.0f}" if severity >= 10 else f"{severity:.1f}"
        direction = "above" if r.get("direction") == "higher" else "below"
        scored.append((severity, f"{r.get('label', r.get('metric'))} {shown}x {direction}"))
    if not scored:
        return ""
    scored.sort(reverse=True)
    parts = [text for _, text in scored[:limit]]
    joined = parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"
    return f"{joined} the rest of the population"


def _prompt(ev: CaseEvidence) -> str:
    payload = asdict(ev)
    payload["reason_codes_explained"] = _reason_lines(ev.reason_codes)
    return (
        "Write the case narrative for this community. The evidence, as JSON:\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


def narrate(ev: CaseEvidence, use_model: bool | None = None) -> dict:
    """Case narrative, from the model when one is reachable and the template otherwise.

    Returns the narrative plus how it was produced, because an analyst reading a
    generated paragraph should be able to see that it was generated.
    """
    if use_model is None:
        use_model = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not use_model:
        return {
            "narrative": template_narrative(ev),
            "source": "template",
            "model": None,
            "note": "No ANTHROPIC_API_KEY set; deterministic narrative.",
        }

    try:
        import anthropic

        client = anthropic.Anthropic()
        request = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": _prompt(ev)}],
        }

        if MODEL.startswith(_FRONTIER_PREFIXES):
            # A case narrative is short writing over assembled evidence, so it
            # does not need the top of the effort range. And a fraud case
            # description can trip a policy classifier -- without a fallback the
            # request simply stops, and an analyst queue that silently drops
            # narratives is worse than one that never had them.
            response = client.beta.messages.create(
                **request,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        else:
            # Haiku 4.5 rejects both of the above with a 400.
            response = client.messages.create(**request)

        # Only populated on the frontier models; harmless to check elsewhere.
        if getattr(response, "stop_reason", None) == "refusal":
            return {
                "narrative": template_narrative(ev),
                "source": "template",
                "model": None,
                "note": "The model declined; deterministic narrative used instead.",
            }

        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise RuntimeError("empty response")
        return {
            "narrative": text,
            "source": "model",
            "model": response.model,
            "note": "Generated from the evidence block beside it. Figures are not verified by the model.",
        }
    except Exception as exc:  # noqa: BLE001 -- the console must not lose the case
        return {
            "narrative": template_narrative(ev),
            "source": "template",
            "model": None,
            "note": f"Model call failed ({type(exc).__name__}: {exc}); deterministic narrative used.",
        }


def evidence_for_ring(
    community,
    graph,
    propagated: dict | None = None,
    behaviour: dict[str, float] | None = None,
    reason_codes: list[str] | None = None,
    index=None,
) -> CaseEvidence:
    """Assemble a ``CaseEvidence`` from the objects the scorer already holds."""
    from contracts.graph_types import NodeType, node_id

    propagated = propagated or {}
    members = set(community.identity_ids)

    shared: dict[str, int] = {}
    g = graph.g
    for ident in community.identity_ids:
        nid = node_id(NodeType.IDENTITY, ident)
        if nid not in g:
            continue
        for nbr in g.neighbors(nid):
            ntype = str(g.nodes[nbr].get("node_type", ""))
            if ntype in ("identity", "account", NodeType.IDENTITY.value):
                continue
            # Only attributes shared by more than one member say anything.
            if sum(1 for m in g.neighbors(nbr) if m != nid and _key_of(g, m) in members):
                shared[ntype] = shared.get(ntype, 0) + 1

    flagged = {k: v for k, v in propagated.items() if k in members}
    dormant = [p for p in flagged.values() if getattr(p, "dormant", False)]
    seed = next((getattr(p, "source_identity", None) for p in flagged.values()), None)

    paths = [
        {
            "identity": p.identity_id,
            "hops": p.hops,
            "via": " -> ".join(h.via for h in p.path) or "direct",
            "score": round(p.propagated_score, 4),
        }
        for p in sorted(flagged.values(), key=lambda x: -x.propagated_score)[:5]
    ]

    stages: dict[str, list] = {}
    if index is not None:
        full = index.for_ring(community.identity_ids)
        stages = {k: full.get(k, []) for k in ("manufacture", "onboard", "weaponise")}

    return CaseEvidence(
        ring_id=community.community_id,
        ring_size=community.size,
        cohesion=round(community.cohesion, 4),
        suspicion=round(community.suspicion, 4),
        institutions=sorted(community.institutions),
        cross_institution=community.is_cross_institution,
        shared_attributes=shared,
        seed_identity=seed,
        n_flagged=len(flagged),
        n_dormant_flagged=len(dormant),
        evidence_paths=paths,
        behaviour=behaviour or {},
        reason_codes=reason_codes or [],
        stages=stages,
    )


def _key_of(g, nid: str) -> str:
    return str(g.nodes[nid].get("key", ""))


def main() -> None:
    from contracts.decisions import ViewScope
    from detect import ingest
    from detect.graph.build import build
    from detect.graph.communities import detect

    ap = argparse.ArgumentParser(description="Case narrative for one community")
    ap.add_argument("--data", default="data/sloppy")
    ap.add_argument("--ring", default=None, help="community id; defaults to the top-scoring one")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", action="store_true", help="force the model path")
    ap.add_argument("--no-model", action="store_true", help="force the template path")
    args = ap.parse_args()

    ds = ingest.load(args.data, view=ViewScope.NETWORK, limit=args.limit)
    print(ds.summary())
    ig = build(ds.onboarding, ds.auth, ds.telemetry)
    communities = detect(ig)
    if not communities:
        raise SystemExit("no communities detected")

    community = (
        next((c for c in communities if c.community_id == args.ring), None)
        if args.ring
        else communities[0]
    )
    if community is None:
        raise SystemExit(f"no community {args.ring}")

    # Confirm the first member that actually transacted, so the narrative has a
    # propagation to describe -- which is the case an analyst would be handed.
    from detect.fusion import mark_dormant, propagate

    transacted = {e.identity_id for e in ds.auth}
    seed = next((i for i in community.identity_ids if i in transacted), None)
    prop = mark_dormant(propagate(ig, {seed: 1.0}), transacted) if seed else {}

    from detect.evidence import EvidenceIndex

    index = EvidenceIndex(ds.onboarding, ds.auth, getattr(ig, "links", []))
    ev = evidence_for_ring(community, ig, prop, index=index)
    use_model = True if args.model else (False if args.no_model else None)
    result = narrate(ev, use_model=use_model)

    print(f"\n--- case narrative for {ev.ring_id} [{result['source']}] ---\n")
    print(result["narrative"])
    print(f"\n({result['note']})")


if __name__ == "__main__":
    main()

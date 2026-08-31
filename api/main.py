"""FastAPI scoring service.

The auth path is deliberately thin: look up precomputed window features, run one
gradient-boosted model, apply the decision bands, return. No graph traversal, no
image work, and emphatically no LLM -- the whole authorisation round trip is a
few hundred milliseconds and scoring owns about 50 of them.

Anything expensive happens off the auth path:

* image hashing at onboarding,
* graph construction and community detection in the background,
* retro-propagation when a confirmation arrives, writing raised scores into the
  same lookup the auth path reads.

That last point is what makes the ring mechanism affordable: the propagation
runs once per confirmation, not once per authorisation.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from contracts.decisions import Action, ScoreResponse, ViewScope
from contracts.schemas import AuthEvent, OnboardingEvent
from detect.decision import DecisionEngine, honeypot_response

DATA_DIR = Path(os.environ.get("FRAUD_DATA", "data/sloppy"))
MODEL_DIR = Path(os.environ.get("FRAUD_MODELS", "models")) / DATA_DIR.name


class ServiceState:
    """Everything loaded once at startup and shared across requests."""

    def __init__(self) -> None:
        self.ready = False
        self.data_dir = DATA_DIR
        self.onboarding_model = None
        self.behaviour_model = None
        self.engine = DecisionEngine()
        self.store = None
        self.graph = None
        self.communities: list = []
        self.ring_of_identity: dict[str, str] = {}
        self.onboarding_score: dict[str, float] = {}
        #: Scores raised by retro-propagation, keyed by identity.
        self.propagated: dict = {}
        #: Recent decisions, so ``/explain`` can answer for an event the
        #: console has just seen go past on the stream. Bounded, because this
        #: is a demo cache and not an audit log -- the real one is the
        #: reason codes written alongside the decision downstream.
        self.decisions: dict = {}
        #: Ring evidence, built alongside the graph. Off the auth path.
        self.evidence = None
        self.latencies: list[float] = []
        self.meta: dict = {}

    def load(self) -> None:
        from detect.features.stream import StreamFeatureStore
        from detect.models.train import TrainedModel

        meta_path = self.data_dir / "meta.json"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        for name in ("onboarding", "behaviour"):
            path = MODEL_DIR / f"{name}.pkl"
            if path.exists():
                setattr(self, f"{name}_model", TrainedModel.load(path))

        self.store = StreamFeatureStore()
        self.ready = True

    def record_decision(self, response) -> None:
        self.decisions[response.event_id] = response
        if len(self.decisions) > 5000:
            for k in list(self.decisions)[:2500]:
                del self.decisions[k]

    def record_latency(self, ms: float) -> None:
        self.latencies.append(ms)
        if len(self.latencies) > 20000:
            del self.latencies[:10000]

    def latency_percentiles(self) -> dict:
        if not self.latencies:
            return {}
        import numpy as np

        a = np.array(self.latencies)
        return {
            "n": int(a.size),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()),
        }


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="Synthetic Identity & Card Testing Detection",
    version="1.0.0",
    description=(
        "Scores the seam between how an account was born and how it behaves. "
        "Auth-path scoring is feature-lookup plus one gradient-boosted model."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state.ready else "loading",
        "scenario": state.meta.get("scenario"),
        "models": {
            "onboarding": state.onboarding_model is not None,
            "behaviour": state.behaviour_model is not None,
        },
        "graph_loaded": state.graph is not None,
        "communities": len(state.communities),
    }


@app.post("/score/auth", response_model=ScoreResponse)
def score_auth(event: AuthEvent, view: ViewScope = ViewScope.NETWORK) -> ScoreResponse:
    """Score one authorisation attempt. This is the latency-critical path."""
    t0 = time.perf_counter()
    if state.store is None or state.behaviour_model is None:
        raise HTTPException(503, "behaviour model not loaded")

    features = state.store.features(event)
    score = float(state.behaviour_model.predict([features])[0])

    onb = state.onboarding_score.get(event.identity_id, 0.0)
    prop = state.propagated.get(event.identity_id)
    if prop is not None:
        from detect.fusion import fuse_with_propagation

        score = fuse_with_propagation({event.identity_id: score}, {event.identity_id: prop})[
            event.identity_id
        ]

    from detect.fusion import fuse_forward

    fused = fuse_forward(score, onb)

    resp = state.engine.decide(
        event_id=event.event_id,
        score=fused,
        features=features,
        onboarding_score=onb,
        ring_id=state.ring_of_identity.get(event.identity_id),
        propagated=prop is not None,
        evidence_path=prop.path if prop is not None else [],
        view=view,
        latency_ms=0.0,
    )
    state.store.update(event)

    ms = (time.perf_counter() - t0) * 1000.0
    state.record_latency(ms)
    out = resp.model_copy(update={"latency_ms": ms})
    state.record_decision(out)
    return out


@app.post("/score/onboarding", response_model=ScoreResponse)
def score_onboarding(event: OnboardingEvent, view: ViewScope = ViewScope.NETWORK) -> ScoreResponse:
    """Score an application at t=0, before the account has done anything."""
    t0 = time.perf_counter()
    if state.onboarding_model is None:
        raise HTTPException(503, "onboarding model not loaded")

    from detect.features.onboarding import OnboardingFeatureBuilder

    builder = OnboardingFeatureBuilder([event])
    score = float(state.onboarding_model.predict([builder.row(event)])[0])
    state.onboarding_score[event.identity_id] = score

    ms = (time.perf_counter() - t0) * 1000.0
    resp = state.engine.decide(
        event_id=event.event_id,
        score=score,
        features={},
        onboarding_score=score,
        ring_id=state.ring_of_identity.get(event.identity_id),
        view=view,
        latency_ms=ms,
    )
    state.record_decision(resp)
    return resp


@app.post("/honeypot")
def honeypot(card_token: str) -> dict:
    """A plausible authorisation result that tells the attacker nothing."""
    return honeypot_response(card_token)


@app.get("/graph/ring/{ring_id}")
def get_ring(ring_id: str, limit: int = 200) -> dict:
    """Nodes and edges for one community, for the console's graph view."""
    if state.graph is None:
        raise HTTPException(503, "graph not built; POST /admin/build-graph first")
    from contracts.graph_types import NodeType, node_id

    community = next((c for c in state.communities if c.community_id == ring_id), None)
    if community is None:
        raise HTTPException(404, f"no community {ring_id}")

    g = state.graph.g
    members = community.identity_ids[:limit]
    member_nodes = {node_id(NodeType.IDENTITY, m) for m in members}

    nodes, edges = [], []
    seen: set[str] = set()
    for m in member_nodes:
        if m not in g:
            continue
        for nid in (m, *g.neighbors(m)):
            if nid in seen:
                continue
            seen.add(nid)
            d = g.nodes[nid]
            nodes.append(
                {
                    "id": nid,
                    "type": d.get("node_type"),
                    "key": d.get("key"),
                    "institutions": sorted(d.get("institutions", ())),
                    "is_member": nid in member_nodes,
                    "propagated": d.get("key") in state.propagated,
                }
            )
    for a, b, d in g.subgraph(seen).edges(data=True):
        edges.append(
            {"source": a, "target": b, "type": d.get("edge_type"), "weight": float(d.get("weight", 1))}
        )

    return {
        "ring_id": ring_id,
        "size": community.size,
        "cohesion": community.cohesion,
        "suspicion": community.suspicion,
        "score": community.score,
        "cross_institution": community.is_cross_institution,
        "institutions": sorted(community.institutions),
        "nodes": nodes,
        "edges": edges,
    }


@app.get("/communities")
def list_communities(top: int = 20) -> list[dict]:
    return [
        {
            "ring_id": c.community_id,
            "size": c.size,
            "cohesion": round(c.cohesion, 4),
            "suspicion": round(c.suspicion, 4),
            "score": round(c.score, 4),
            "cross_institution": c.is_cross_institution,
            "institutions": sorted(c.institutions),
        }
        for c in state.communities[:top]
    ]


@app.post("/confirm/{identity_id}")
def confirm_fraud(identity_id: str, decay: float = 0.55, max_hops: int = 3) -> dict:
    """Confirm an account as fraudulent and retro-propagate to its ring.

    This is the demo. One confirmation, and every sibling that shares
    infrastructure lights up -- including accounts that have never transacted
    and therefore cannot be caught any other way.
    """
    if state.graph is None:
        raise HTTPException(503, "graph not built; POST /admin/build-graph first")
    from detect.fusion import mark_dormant, propagate

    prop = propagate(state.graph, {identity_id: 1.0}, decay=decay, max_hops=max_hops)
    transacted = getattr(state, "transacted_identities", set())
    prop = mark_dormant(prop, transacted)
    state.propagated.update(prop)

    dormant = [p for p in prop.values() if p.dormant]
    return {
        "seed": identity_id,
        "n_flagged": len(prop),
        "n_dormant_flagged": len(dormant),
        "flagged": [
            {
                "identity_id": p.identity_id,
                "score": round(p.propagated_score, 4),
                "hops": p.hops,
                "dormant": p.dormant,
                "path": [h.model_dump() for h in p.path],
            }
            for p in sorted(prop.values(), key=lambda x: -x.propagated_score)[:200]
        ],
    }


@app.get("/explain/{event_id}")
def explain(event_id: str) -> dict:
    """Human-readable reasons behind a decision."""
    cached = state.decisions.get(event_id)
    if cached is None:
        raise HTTPException(404, f"no decision recorded for {event_id}")
    return {
        "event_id": event_id,
        "score": cached.score,
        "band": cached.band.value,
        "action": cached.action.value,
        "propagated": cached.propagated,
        "reasons": cached.explain(),
        "evidence_path": [h.model_dump() for h in cached.evidence_path],
    }


@app.get("/ring/{ring_id}/evidence")
def ring_evidence(ring_id: str) -> dict:
    """Why this community is a ring, as the three stages of the supply chain.

    Manufacture, onboard, weaponise -- each figure beside the same figure for
    the legitimate population, because a decline ratio of 0.82 means nothing
    until it sits next to a population ratio of 0.04.
    """
    if state.evidence is None:
        raise HTTPException(503, "evidence index not built; POST /admin/build-graph first")

    community = next((c for c in state.communities if c.community_id == ring_id), None)
    if community is None:
        raise HTTPException(404, f"no community {ring_id}")

    payload = state.evidence.for_ring(community.identity_ids)
    payload["ring_id"] = ring_id
    payload["cross_institution"] = community.is_cross_institution
    payload["institutions"] = sorted(community.institutions)
    return payload


@app.post("/narrate/{ring_id}")
def narrate_ring(ring_id: str, use_model: bool | None = None) -> dict:
    """A case narrative for one community. Deliberately off the auth path.

    An LLM in the authorisation round trip would blow the latency budget by two
    orders of magnitude. Here it reads a subgraph that has already been scored
    and writes the paragraph an analyst would otherwise write by hand -- and it
    sees only tokens, counts and scores, never PII. With no API key configured
    the deterministic narrative is returned instead, so the console works the
    same either way.
    """
    if state.graph is None:
        raise HTTPException(503, "graph not built; POST /admin/build-graph first")
    from detect.copilot import evidence_for_ring, narrate

    community = next((c for c in state.communities if c.community_id == ring_id), None)
    if community is None:
        raise HTTPException(404, f"no community {ring_id}")

    ev = evidence_for_ring(community, state.graph, state.propagated)
    result = narrate(ev, use_model=use_model)
    result["evidence"] = ev.__dict__
    result["ring_id"] = ring_id
    return result


@app.get("/metrics")
def metrics() -> dict:
    out = {
        "scenario": state.meta.get("scenario"),
        "latency": state.latency_percentiles(),
        "propagated_identities": len(state.propagated),
        "communities": len(state.communities),
    }
    mpath = MODEL_DIR / "metrics.json"
    if mpath.exists():
        out["models"] = json.loads(mpath.read_text(encoding="utf-8"))
    return out


@app.post("/admin/build-graph")
def build_graph(limit: int | None = None) -> dict:
    """Build the identity graph and detect communities. Off the auth path."""
    from detect import ingest
    from detect.graph.build import build
    from detect.graph.communities import detect

    ds = ingest.load(state.data_dir, view=ViewScope.NETWORK, limit=limit)
    state.graph = build(ds.onboarding, ds.auth, ds.telemetry)
    state.communities = detect(state.graph)
    state.ring_of_identity = {
        ident: c.community_id for c in state.communities for ident in c.identity_ids
    }
    state.transacted_identities = {e.identity_id for e in ds.auth}

    from detect.evidence import EvidenceIndex

    state.evidence = EvidenceIndex(ds.onboarding, ds.auth, getattr(state.graph, "links", []))
    return {
        "graph": state.graph.summary(),
        "communities": len(state.communities),
        "transacted_identities": len(state.transacted_identities),
        "evidence_index": len(state.evidence.onboarding),
    }


@app.get("/stream")
def stream(
    speed: float = 200.0,
    limit: int = 2000,
    view: ViewScope = ViewScope.NETWORK,
    institution_id: str | None = None,
):
    """Replay the scenario as a live event stream for the console.

    A merchant-scoped replay needs to know *which* merchant: the scoping is
    done at read time, so without an institution there is no slice to read.
    Default to the first institution in the scenario rather than failing, so
    the console's toggle works without the operator having to pick one.
    """
    from detect import ingest

    if view is not ViewScope.NETWORK and institution_id is None:
        found = ingest.institutions_in(state.data_dir)
        if not found:
            raise HTTPException(503, "no institutions in this scenario")
        institution_id = found[0]

    def gen():
        ds = ingest.load(state.data_dir, view=view, institution_id=institution_id, limit=limit)
        for ev in ds.auth[:limit]:
            resp = score_auth(ev, view=view)
            payload = {
                "event": {
                    "event_id": ev.event_id,
                    "ts": ev.ts.isoformat(),
                    "account_id": ev.account_id,
                    "identity_id": ev.identity_id,
                    "merchant_id": ev.merchant_id,
                    "institution_id": ev.institution_id,
                    "amount": ev.amount,
                    "approved": ev.approved,
                },
                "scope": {"view": view.value, "institution_id": institution_id},
                "decision": json.loads(resp.model_dump_json()),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1.0 / max(1.0, speed))

    return StreamingResponse(gen(), media_type="text/event-stream")

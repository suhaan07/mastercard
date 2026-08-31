"""L5: fusion and retro-propagation. The contribution.

Two directions, and the second is the one worth building.

**Forward.** An account's onboarding score conditions its behavioural
thresholds. A borderline-onboarded account testing cards should trip far sooner
than a ten-year customer doing the same thing. Today these two scores live in
different systems and never meet, which is the seam the design doc identifies.

**Backward -- retro-propagation.** When an account is confirmed bad, its
evidence diffuses back through the identity graph to its ring siblings,
*including dormant accounts that have done nothing at all*. Those accounts have
no behavioural signal to detect, by construction: they have not transacted. The
only way to reach them is through what they share with an account that has.

Implemented as weighted score diffusion with per-hop decay rather than as a GNN.
That is a deliberate choice, not a shortcut:

* Every propagated score carries the **path** that produced it, so the analyst
  console can say "raised because it shares a device with an account confirmed
  fraudulent two hops away" instead of "the network said 0.87". Explainability
  is on the design doc's constraints table.
* It needs no training data about rings, so it works on a ring shape never seen
  before -- which is what an adapting adversary produces.
* It is cheap enough to run inside the auth path.
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field

from contracts.decisions import EvidenceHop
from contracts.graph_types import NodeType, node_id, parse_node_id
from detect.graph.build import IdentityGraph

#: How much evidence survives each hop. Two accounts on one device are strongly
#: related; two accounts three hops apart through a shared ISP are barely
#: related at all, and the decay has to say so.
DEFAULT_DECAY = 0.55
#: Beyond three hops almost everything is connected to everything, so
#: propagation past this point spreads suspicion rather than evidence.
DEFAULT_MAX_HOPS = 3
#: Below this, a propagated contribution is not worth carrying.
MIN_CONTRIBUTION = 0.02

#: Edge types that carry evidence between identities, and how well.
#: A shared merchant carries none -- everyone shops somewhere.
EDGE_CONDUCTANCE: dict[str, float] = {
    "similar_tag": 1.0,
    "pii_recombination": 1.0,
    "observed_together": 0.8,
    "owns": 0.95,
    "transacted": 0.15,
}

#: Attribute nodes evidence may flow through, and how well each conducts.
NODE_CONDUCTANCE: dict[str, float] = {
    NodeType.DEVICE.value: 1.0,
    NodeType.ADDRESS.value: 0.7,
    NodeType.PHONE.value: 0.65,
    NodeType.EMAIL.value: 0.6,
    NodeType.IP_ASN.value: 0.35,
    NodeType.CARD_TOKEN.value: 0.25,
    NodeType.ACCOUNT.value: 0.9,
    NodeType.IDENTITY.value: 0.9,
    # A shared merchant is not evidence of anything.
    NodeType.MERCHANT.value: 0.0,
}


@dataclass
class Propagated:
    """A score raised by evidence from elsewhere, with its justification."""

    identity_id: str
    propagated_score: float
    hops: int
    source_identity: str
    path: list[EvidenceHop] = field(default_factory=list)
    #: True when this identity has never transacted. These are the accounts the
    #: whole mechanism exists for -- nothing else can reach them.
    dormant: bool = False


def _conductance(g, a: str, b: str) -> float:
    """How well evidence flows along one edge, combining edge and node type."""
    data = g[a][b]
    edge_c = EDGE_CONDUCTANCE.get(str(data.get("edge_type", "observed_together")), 0.5)
    node_c = NODE_CONDUCTANCE.get(str(g.nodes[b].get("node_type", "")), 0.5)
    # Weight accumulates with repeat observation; compress it so that a device
    # seen 400 times does not dominate one seen 5 times by 80x.
    w = float(data.get("weight", 1.0))
    strength = min(1.0, w / (1.0 + w))
    return edge_c * node_c * (0.5 + 0.5 * strength)


def propagate(
    ig: IdentityGraph,
    seeds: dict[str, float],
    decay: float = DEFAULT_DECAY,
    max_hops: int = DEFAULT_MAX_HOPS,
    min_contribution: float = MIN_CONTRIBUTION,
) -> dict[str, Propagated]:
    """Diffuse confirmed-bad evidence from ``seeds`` through the graph.

    ``seeds`` maps identity_id to the strength of the confirmation (1.0 for a
    confirmed fraud, lower for a strong suspicion). Returns the identities whose
    scores were raised, each with the path that raised it.

    Best-first traversal: the strongest evidence is expanded first, so the
    surviving path to each identity is the strongest one rather than merely the
    shortest.
    """
    g = ig.g
    best: dict[str, Propagated] = {}

    # Max-heap by contribution (negated, since heapq is a min-heap). The
    # counter is a tiebreaker: without it, two entries with the same score and
    # node fall through to comparing the evidence paths, and heapq raises on
    # the first pair of dicts it reaches.
    counter = itertools.count()
    frontier: list[tuple[float, int, str, int, str, tuple]] = []
    for ident, strength in seeds.items():
        nid = node_id(NodeType.IDENTITY, ident)
        if nid in g:
            heapq.heappush(frontier, (-strength, next(counter), nid, 0, ident, ()))

    visited: dict[str, float] = {}
    while frontier:
        neg_score, _, nid, hops, source, path = heapq.heappop(frontier)
        score = -neg_score
        if score < min_contribution or hops > max_hops:
            continue
        if visited.get(nid, 0.0) >= score:
            continue
        visited[nid] = score

        ntype, key = parse_node_id(nid)
        if ntype is NodeType.IDENTITY and hops > 0 and key not in seeds:
            prev = best.get(key)
            if prev is None or score > prev.propagated_score:
                best[key] = Propagated(
                    identity_id=key,
                    propagated_score=score,
                    hops=hops,
                    source_identity=source,
                    path=[EvidenceHop(**h) for h in path],
                )

        if hops >= max_hops:
            continue

        for nbr in g.neighbors(nid):
            c = _conductance(g, nid, nbr)
            if c <= 0.0:
                continue
            nxt = score * decay * c
            if nxt < min_contribution:
                continue
            if visited.get(nbr, 0.0) >= nxt:
                continue
            hop = {
                "from_node": nid,
                "to_node": nbr,
                "via": str(g[nid][nbr].get("edge_type", "observed_together")),
                "contribution": float(nxt),
            }
            heapq.heappush(
                frontier, (-nxt, next(counter), nbr, hops + 1, source, path + (hop,))
            )

    return best


def mark_dormant(
    propagated: dict[str, Propagated], transacted_identities: set[str]
) -> dict[str, Propagated]:
    """Flag propagated identities that have never transacted.

    This is the number the demo turns on: how many accounts were flagged
    *before* doing anything. Counting them is the difference between "we
    detected a ring" and "we detected a ring before it acted".
    """
    for ident, p in propagated.items():
        p.dormant = ident not in transacted_identities
    return propagated


def fuse_forward(
    behaviour_score: float,
    onboarding_score: float,
    weight: float = 0.35,
) -> float:
    """Condition a behavioural score on how the account was born.

    A multiplicative lift rather than an average: a clean-onboarding account
    behaving badly should still be caught on behaviour alone, so onboarding can
    only *raise* the score, never suppress it. Averaging would let a
    well-manufactured identity dilute genuine behavioural evidence, which is
    exactly backwards.
    """
    lift = 1.0 + weight * onboarding_score
    return float(min(1.0, behaviour_score * lift))


def fuse_with_propagation(
    base_scores: dict[str, float],
    propagated: dict[str, Propagated],
    weight: float = 0.7,
) -> dict[str, float]:
    """Combine each identity's own score with evidence propagated to it.

    Takes the stronger of the two rather than summing, so a single confirmed
    sibling cannot push an otherwise-clean account past a block threshold on its
    own -- the propagated score is capped by ``weight``.
    """
    out = dict(base_scores)
    for ident, p in propagated.items():
        out[ident] = max(out.get(ident, 0.0), weight * p.propagated_score)
    return out


def ring_alert(
    ig: IdentityGraph,
    confirmed_identity: str,
    transacted_identities: set[str],
    decay: float = DEFAULT_DECAY,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> dict:
    """The stage demo, as one call.

    One account is confirmed fraudulent; report every sibling that lights up and
    how many of them have never transacted.
    """
    prop = propagate(ig, {confirmed_identity: 1.0}, decay=decay, max_hops=max_hops)
    prop = mark_dormant(prop, transacted_identities)
    dormant = [p for p in prop.values() if p.dormant]
    by_hop: dict[int, int] = defaultdict(int)
    for p in prop.values():
        by_hop[p.hops] += 1
    return {
        "seed": confirmed_identity,
        "n_flagged": len(prop),
        "n_dormant_flagged": len(dormant),
        "by_hop": dict(sorted(by_hop.items())),
        "flagged": sorted(prop.values(), key=lambda p: -p.propagated_score),
    }

"""L2: community detection over the identity graph, proposing candidate rings.

Louvain finds densely-connected groups, but density alone is not suspicion --
four housemates share an address and a router, and that is a dense community of
entirely innocent people. So every community gets a **cohesion x suspicion**
score: how tightly connected it is, multiplied by how much of that connection
comes from attributes that legitimate people do not share.

Run as a script to evaluate recovery against planted rings::

    python -m detect.graph.communities --data data/sloppy
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from contracts.graph_types import SUSPICIOUS_SHARE_WEIGHT, NodeType, parse_node_id
from detect.graph.build import IdentityGraph

#: Communities smaller than this are not rings, they are households.
MIN_RING_SIZE = 4
#: Attribute types whose sharing carries real ring signal.
RING_SIGNAL_TYPES = (
    NodeType.DEVICE,
    NodeType.ADDRESS,
    NodeType.PHONE,
    NodeType.IP_ASN,
    NodeType.EMAIL,
)
_RING_SIGNAL_VALUES = {t.value for t in RING_SIGNAL_TYPES}


@dataclass
class Community:
    """A proposed ring."""

    community_id: str
    identity_ids: list[str]
    #: Mean edge weight among members: how strongly the group is bound.
    cohesion: float
    #: Share of that binding coming from attributes legitimate people do not share.
    suspicion: float
    #: Reported as a diagnostic, deliberately *not* part of the score. See below.
    density: float
    shared_attributes: dict[str, int]
    institutions: set[str]

    @property
    def score(self) -> float:
        """Cohesion x suspicion x log(size).

        The size term is not decoration. Scoring on edge *density* -- the
        obvious choice -- ranked every planted ring between 104th and 119th out
        of 130 communities, because density falls off quadratically with size:
        a household of four sharing an address is nearly complete and scores
        ~0.6, while a genuine 51-account ring is a sparse cluster and scores
        near zero. Density measures the wrong thing here.

        What actually distinguishes a ring is that *many* identities are bound
        by *suspicious* infrastructure. Weighting by log(size) moved all four
        planted rings into the top nine and put every synthetic identity in the
        top ten communities.
        """
        return self.cohesion * self.suspicion * math.log1p(self.size)

    @property
    def size(self) -> int:
        return len(self.identity_ids)

    @property
    def is_cross_institution(self) -> bool:
        """True when the group spans institutions.

        These are the groups a single merchant or issuer *cannot* see, and they
        are the entire argument for network-level deployment.
        """
        return len(self.institutions) > 1


def _identity_projection(ig: IdentityGraph) -> nx.Graph:
    """Project the bipartite graph onto identities alone.

    Two identities are connected when they share infrastructure, weighted by how
    surprising that sharing is and inversely by how many others share the same
    thing. Louvain needs a one-mode graph; run on the bipartite structure it
    would cluster devices and merchants alongside people.
    """
    proj = nx.Graph()
    g = ig.g

    for nid, data in g.nodes(data=True):
        if data.get("node_type") != NodeType.IDENTITY.value:
            continue
        proj.add_node(nid, node_type=data.get("node_type"), key=data.get("key"))
        proj.nodes[nid]["institutions"] = set(data.get("institutions", ()))

    def _bump(a: str, b: str, w: float, tag: str) -> None:
        if proj.has_edge(a, b):
            proj[a][b]["weight"] += w
            proj[a][b]["shared"][tag] += 1
        else:
            proj.add_edge(a, b, weight=w, shared=Counter({tag: 1}))

    # Shared-attribute paths: identity -> attribute -> identity.
    for nid, data in g.nodes(data=True):
        ntype = data.get("node_type")
        if ntype not in _RING_SIGNAL_VALUES:
            continue
        owners = [
            n for n in g.neighbors(nid) if g.nodes[n].get("node_type") == NodeType.IDENTITY.value
        ]
        if not 2 <= len(owners) <= 60:
            continue
        base = SUSPICIOUS_SHARE_WEIGHT[NodeType(ntype)]
        # Inverse-frequency: an attribute shared by fifty identities says far
        # less per pair than one shared by three.
        w = base / (len(owners) - 1) ** 0.5
        for i, a in enumerate(owners):
            for b in owners[i + 1 :]:
                _bump(a, b, w, ntype)

    # Direct identity-to-identity edges from entity resolution.
    for a, b, data in g.edges(data=True):
        if (
            g.nodes[a].get("node_type") == NodeType.IDENTITY.value
            and g.nodes[b].get("node_type") == NodeType.IDENTITY.value
        ):
            _bump(a, b, float(data.get("weight", 1.0)), str(data.get("edge_type", "link")))

    return proj


def detect(ig: IdentityGraph, resolution: float = 1.0, seed: int = 7) -> list[Community]:
    """Find candidate rings and score them."""
    proj = _identity_projection(ig)
    if proj.number_of_edges() == 0:
        return []

    # networkx 3.6 ships Louvain, so there is no python-louvain dependency.
    partitions = nx.community.louvain_communities(
        proj, weight="weight", resolution=resolution, seed=seed
    )

    out: list[Community] = []
    for i, part in enumerate(sorted(partitions, key=len, reverse=True)):
        members = [m for m in part if m in proj]
        if len(members) < MIN_RING_SIZE:
            continue
        sub = proj.subgraph(members)

        possible = len(members) * (len(members) - 1) / 2
        n_edges = sub.number_of_edges()
        density = n_edges / possible if possible else 0.0
        cohesion = float(
            sum(d["weight"] for _, _, d in sub.edges(data=True)) / n_edges if n_edges else 0.0
        )

        shared: Counter = Counter()
        for _, _, d in sub.edges(data=True):
            shared.update(d.get("shared", {}))
        total_shared = sum(shared.values()) or 1
        # Suspicion is the share of connective tissue coming from things
        # legitimate people do not have in common.
        suspicious_mass = sum(
            c
            for k, c in shared.items()
            if k in _RING_SIGNAL_VALUES or k in ("similar_tag", "pii_recombination")
        )
        suspicion = float(suspicious_mass / total_shared)

        institutions: set[str] = set()
        for m in members:
            institutions |= set(proj.nodes[m].get("institutions", ()))

        out.append(
            Community(
                community_id=f"cmt_{i:04d}",
                identity_ids=[parse_node_id(m)[1] for m in members],
                cohesion=cohesion,
                suspicion=suspicion,
                density=float(density),
                shared_attributes=dict(shared),
                institutions=institutions,
            )
        )

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def evaluate(communities: list[Community], truth_by_identity: dict[str, str | None]) -> dict:
    """Score recovered communities against planted rings.

    Reports, per planted ring, the best-matching community's recall and
    precision -- the honest way to say "we found the ring" rather than gesturing
    at a picture of a graph.
    """
    planted: dict[str, set[str]] = defaultdict(set)
    for ident, ring in truth_by_identity.items():
        if ring:
            planted[ring].add(ident)

    index = {c.community_id: r for r, c in enumerate(communities)}
    results = []
    for ring_id, members in sorted(planted.items()):
        best: dict | None = None
        for c in communities:
            found = set(c.identity_ids)
            hit = len(found & members)
            if not hit:
                continue
            recall = hit / len(members)
            precision = hit / len(found)
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
            if best is None or f1 > best["f1"]:
                best = {
                    "ring_id": ring_id,
                    "ring_size": len(members),
                    "community_id": c.community_id,
                    "community_size": c.size,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "rank": index[c.community_id],
                }
        results.append(
            best
            or {
                "ring_id": ring_id,
                "ring_size": len(members),
                "community_id": "-",
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "rank": -1,
            }
        )

    mean_recall = sum(r["recall"] for r in results) / max(1, len(results))
    mean_f1 = sum(r["f1"] for r in results) / max(1, len(results))
    return {"per_ring": results, "mean_recall": mean_recall, "mean_f1": mean_f1}


def main() -> None:
    from contracts.decisions import ViewScope
    from detect import ingest
    from detect.graph.build import build

    ap = argparse.ArgumentParser(description="Detect candidate rings")
    ap.add_argument("--data", default="data/sloppy")
    ap.add_argument("--limit", type=int, default=None, help="cap events for a fast pass")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    ds = ingest.load(args.data, view=ViewScope.NETWORK, limit=args.limit)
    print(ds.summary())

    ig = build(ds.onboarding, ds.auth, ds.telemetry)
    print(f"graph: {ig.summary()}")
    print(f"entity-resolution links: {len(ig.links):,}")

    communities = detect(ig)
    print(f"\ncommunities >= {MIN_RING_SIZE}: {len(communities)}")

    truth = {t.identity_id: t.ring_id for t in ds.ground_truth}
    synth = {t.identity_id for t in ds.ground_truth if t.is_synthetic}

    header = (
        f"{'rank':>4} {'id':<10} {'size':>5} {'cohesion':>9} "
        f"{'suspicion':>10} {'density':>8} {'score':>8}  synthetic  x-inst"
    )
    print("\n" + header)
    for i, c in enumerate(communities[: args.top]):
        n_syn = sum(1 for x in c.identity_ids if x in synth)
        frac = f"{n_syn}/{c.size}"
        print(
            f"{i:>4} {c.community_id:<10} {c.size:>5} {c.cohesion:>9.3f} "
            f"{c.suspicion:>10.3f} {c.density:>8.3f} {c.score:>8.3f} {frac:>10}  {c.is_cross_institution}"
        )

    ev = evaluate(communities, truth)
    print(f"\nring recovery: mean recall {ev['mean_recall']:.3f}, mean F1 {ev['mean_f1']:.3f}")
    print(f"{'ring':<12}{'size':>6}{'community':>12}{'recall':>9}{'precision':>11}{'F1':>8}{'rank':>6}")
    for r in ev["per_ring"]:
        print(
            f"{r['ring_id']:<12}{r['ring_size']:>6}{r['community_id']:>12}"
            f"{r['recall']:>9.3f}{r['precision']:>11.3f}{r['f1']:>8.3f}{r['rank']:>6}"
        )


if __name__ == "__main__":
    main()

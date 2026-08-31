"""L2: build the identity graph from the event streams.

Nodes are identities and the infrastructure they touch; edges record that two
things were observed together, weighted by how *surprising* the sharing is. That
weighting is the whole game. Two accounts sharing a merchant is meaningless --
half the population shops there. Two accounts sharing a device fingerprint is
close to conclusive. ``SUSPICIOUS_SHARE_WEIGHT`` in ``contracts/graph_types.py``
holds those priors in one place.

The graph is the substrate for two things: community detection (proposing rings)
and retro-propagation (diffusing confirmed-bad evidence to siblings). Both need
the same structure, which is why this builds once and is reused.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from contracts.graph_types import SUSPICIOUS_SHARE_WEIGHT, EdgeType, NodeType, node_id, parse_node_id
from contracts.schemas import AuthEvent, OnboardingEvent, SessionTelemetry
from detect.graph.entity_res import Link, resolve


@dataclass
class IdentityGraph:
    """The graph plus the lookups the rest of the pipeline needs."""

    g: nx.Graph
    #: identity_id -> account_id and back, since events key on both.
    account_of: dict[str, str] = field(default_factory=dict)
    identity_of: dict[str, str] = field(default_factory=dict)
    #: Institutions that have observed each node. Length > 1 is only visible in
    #: the network view, and is what the view toggle acts on.
    institutions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    links: list[Link] = field(default_factory=list)

    def identity_nodes(self) -> list[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("node_type") == NodeType.IDENTITY.value]

    def neighbours_of_type(self, nid: str, node_type: NodeType) -> list[str]:
        return [
            n
            for n in self.g.neighbors(nid)
            if self.g.nodes[n].get("node_type") == node_type.value
        ]

    def summary(self) -> str:
        by_type: dict[str, int] = defaultdict(int)
        for _, d in self.g.nodes(data=True):
            by_type[d.get("node_type", "?")] += 1
        types = " ".join(f"{k}={v:,}" for k, v in sorted(by_type.items()))
        return f"{self.g.number_of_nodes():,} nodes ({types}), {self.g.number_of_edges():,} edges"


def _add_node(g: nx.Graph, node_type: NodeType, key: str, institution: str | None = None) -> str:
    nid = node_id(node_type, key)
    if nid not in g:
        g.add_node(nid, node_type=node_type.value, key=key, institutions=set())
    if institution:
        g.nodes[nid]["institutions"].add(institution)
    return nid


def _add_edge(
    g: nx.Graph, a: str, b: str, edge_type: EdgeType, weight: float, institution: str
) -> None:
    """Accumulate weight on repeat observation rather than overwriting.

    Seeing the same account on the same device forty times is stronger evidence
    than seeing it once, but not forty times stronger -- so weight accumulates
    and the caller applies a sublinear transform when scoring cohesion.
    """
    if g.has_edge(a, b):
        e = g[a][b]
        e["weight"] += weight
        e["count"] += 1
        e["institutions"].add(institution)
    else:
        g.add_edge(
            a,
            b,
            weight=weight,
            count=1,
            edge_type=edge_type.value,
            institutions={institution},
        )


def build(
    onboarding: list[OnboardingEvent],
    auth: list[AuthEvent],
    telemetry: list[SessionTelemetry] | None = None,
    include_entity_resolution: bool = True,
) -> IdentityGraph:
    """Construct the identity graph from the three observation streams."""
    g = nx.Graph()
    ig = IdentityGraph(g=g)

    # -- onboarding: identity and the infrastructure it arrived on ---------
    for ev in onboarding:
        inst = ev.institution_id
        ident = _add_node(g, NodeType.IDENTITY, ev.identity_id, inst)
        acct = _add_node(g, NodeType.ACCOUNT, ev.account_id, inst)
        ig.account_of[ev.identity_id] = ev.account_id
        ig.identity_of[ev.account_id] = ev.identity_id
        _add_edge(g, ident, acct, EdgeType.OWNS, 1.0, inst)

        for node_type, key in (
            (NodeType.DEVICE, ev.device_id),
            (NodeType.IP_ASN, ev.ip_id),
            (NodeType.EMAIL, ev.email_token),
            (NodeType.PHONE, ev.phone_token),
            (NodeType.ADDRESS, ev.address_token),
        ):
            n = _add_node(g, node_type, key, inst)
            _add_edge(
                g, ident, n, EdgeType.OBSERVED_TOGETHER, SUSPICIOUS_SHARE_WEIGHT[node_type], inst
            )

    # -- auth: accounts, cards, merchants, and the kit used at the time ----
    for ev in auth:
        inst = ev.institution_id
        acct = _add_node(g, NodeType.ACCOUNT, ev.account_id, inst)
        card = _add_node(g, NodeType.CARD_TOKEN, ev.card_token, inst)
        merch = _add_node(g, NodeType.MERCHANT, ev.merchant_id, inst)
        _add_edge(g, acct, card, EdgeType.TRANSACTED, SUSPICIOUS_SHARE_WEIGHT[NodeType.CARD_TOKEN], inst)
        _add_edge(g, acct, merch, EdgeType.TRANSACTED, SUSPICIOUS_SHARE_WEIGHT[NodeType.MERCHANT], inst)
        dev = _add_node(g, NodeType.DEVICE, ev.device_id, inst)
        ip = _add_node(g, NodeType.IP_ASN, ev.ip_id, inst)
        _add_edge(g, acct, dev, EdgeType.OBSERVED_TOGETHER, SUSPICIOUS_SHARE_WEIGHT[NodeType.DEVICE], inst)
        _add_edge(g, acct, ip, EdgeType.OBSERVED_TOGETHER, SUSPICIOUS_SHARE_WEIGHT[NodeType.IP_ASN], inst)

    for ev in telemetry or []:
        inst = ev.institution_id
        acct = _add_node(g, NodeType.ACCOUNT, ev.account_id, inst)
        dev = _add_node(g, NodeType.DEVICE, ev.device_id, inst)
        _add_edge(g, acct, dev, EdgeType.OBSERVED_TOGETHER, SUSPICIOUS_SHARE_WEIGHT[NodeType.DEVICE], inst)

    # -- entity resolution: direct identity-to-identity links --------------
    if include_entity_resolution:
        ig.links = resolve(onboarding)
        for link in ig.links:
            a = node_id(NodeType.IDENTITY, link.left)
            b = node_id(NodeType.IDENTITY, link.right)
            if a not in g or b not in g:
                continue
            etype = (
                EdgeType.PII_RECOMBINATION
                if link.kind.startswith("pii_recombination")
                else EdgeType.SIMILAR_TAG
                if link.kind.startswith("near_duplicate")
                else EdgeType.OBSERVED_TOGETHER
            )
            inst = next(iter(g.nodes[a]["institutions"]), "unknown")
            _add_edge(g, a, b, etype, link.weight, inst)

    for n, d in g.nodes(data=True):
        ig.institutions[n] = d["institutions"]

    return ig

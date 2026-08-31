"""Identity-graph node and edge types, and the canonical node-id convention.

Every module builds node ids through :func:`node_id`. If two modules disagree
about how to spell a device node, the graph silently splits and ring detection
quietly stops working -- so the spelling lives here and nowhere else.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

NODE_ID_SEP = ":"


class NodeType(str, Enum):
    IDENTITY = "identity"
    ACCOUNT = "account"
    DEVICE = "device"
    IP_ASN = "ip_asn"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CARD_TOKEN = "card_token"
    MERCHANT = "merchant"
    #: A bucket of near-duplicate face tags, not a face. See biohash/flyhash.py.
    FACE_CLUSTER = "face_cluster"
    #: A shared document-generation artifact.
    DOC_TEMPLATE = "doc_template"


class EdgeType(str, Enum):
    #: Generic co-occurrence, weighted by how surprising the sharing is.
    OBSERVED_TOGETHER = "observed_together"
    #: identity -> account
    OWNS = "owns"
    #: account -> card_token, from an auth attempt
    TRANSACTED = "transacted"
    #: identity -> face_cluster / doc_template, from tag overlap
    SIMILAR_TAG = "similar_tag"
    #: identity -> identity, where tokenised PII is reused across names
    PII_RECOMBINATION = "pii_recombination"


class GraphNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_type: NodeType
    key: str
    #: Which institutions have observed this node. Length > 1 is only ever
    #: visible in the network view -- this field is what the merchant/network
    #: toggle filters on.
    institutions: frozenset[str] = frozenset()

    @property
    def id(self) -> str:
        return node_id(self.node_type, self.key)


class GraphEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    src: str
    dst: str
    edge_type: EdgeType
    ts: datetime
    weight: float = Field(default=1.0, ge=0.0)
    institution_id: str


def node_id(node_type: NodeType, key: str) -> str:
    """Canonical node identifier. The only sanctioned way to spell a node."""
    return f"{node_type.value}{NODE_ID_SEP}{key}"


def parse_node_id(nid: str) -> tuple[NodeType, str]:
    """Inverse of :func:`node_id`."""
    head, _, key = nid.partition(NODE_ID_SEP)
    return NodeType(head), key


#: Shared-attribute types that carry ring signal. A shared merchant means
#: nothing; a shared device means a great deal. Weights feed the community
#: cohesion score in detect/graph/communities.py.
SUSPICIOUS_SHARE_WEIGHT: dict[NodeType, float] = {
    NodeType.DEVICE: 1.0,
    NodeType.FACE_CLUSTER: 1.0,
    NodeType.DOC_TEMPLATE: 0.9,
    NodeType.ADDRESS: 0.7,
    NodeType.PHONE: 0.6,
    NodeType.EMAIL: 0.6,
    NodeType.IP_ASN: 0.4,
    NodeType.CARD_TOKEN: 0.3,
    NodeType.MERCHANT: 0.0,
    NodeType.IDENTITY: 0.0,
    NodeType.ACCOUNT: 0.0,
}

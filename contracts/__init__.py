"""Frozen contracts shared by every module.

Phase 0 of the build plan: schemas, graph types and decision types are defined
here once and imported everywhere. No module may redefine them -- if a field is
needed, it is added here first.
"""

from contracts.schemas import (
    SCHEMA_VERSION,
    AuthEvent,
    AuthResponseCode,
    AvsResult,
    CvvResult,
    GroundTruth,
    Label,
    LabelSource,
    OnboardingEvent,
    SessionTelemetry,
    SparseTag,
    VerificationSignals,
)
from contracts.graph_types import EdgeType, GraphEdge, GraphNode, NodeType, node_id, parse_node_id
from contracts.decisions import (
    Action,
    Band,
    ReasonCode,
    ScoreRequest,
    ScoreResponse,
    ViewScope,
    reason_text,
)

__all__ = [
    "SCHEMA_VERSION",
    "AuthEvent",
    "AuthResponseCode",
    "AvsResult",
    "CvvResult",
    "GroundTruth",
    "Label",
    "LabelSource",
    "OnboardingEvent",
    "SessionTelemetry",
    "SparseTag",
    "VerificationSignals",
    "EdgeType",
    "GraphEdge",
    "GraphNode",
    "NodeType",
    "node_id",
    "parse_node_id",
    "Action",
    "Band",
    "ReasonCode",
    "ScoreRequest",
    "ScoreResponse",
    "ViewScope",
    "reason_text",
]

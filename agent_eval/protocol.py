"""Shared protocol types for detector and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectorDecision:
    """Result of a single detection step."""

    sample_id: str
    is_malicious_pred: bool
    risk_score: float
    char_spans_pred: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyDecision:
    """Policy action derived from detector decision."""

    action: str  # "block", "quarantine", "preserve"
    reason: str = ""


@dataclass
class SystemResult:
    """Full system evaluation result for a trajectory."""

    attack_blocked: bool = False
    num_alerts: int = 0
    final_disposition: str = "clean"

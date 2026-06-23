"""Entropy monitoring channel — extracted from DualSentinel's InFliGuard detector.

Monitors per-token entropy during LLM generation to detect anomalous low-entropy
patterns that indicate the model is being steered by injected instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import List, Tuple

import numpy as np


@dataclass
class EntropySignal:
    """Current state of the entropy monitoring channel."""
    mean_entropy: float
    std_entropy: float
    is_suspicious: bool
    confidence: float  # 0.0 ~ 1.0, how anomalous the current window is
    anomaly_score: float  # 0.0 ~ 1.0, how far below threshold


class EntropyMonitor:
    """Sliding-window entropy stability detector (InFliGuard algorithm).

    Detects consecutive windows of low, stable entropy — a signature of
    model being controlled by injected instructions.
    """

    def __init__(
        self,
        window_size: int = 5,
        mean_threshold: float = 1e-2,
        consecutive_threshold: int = 6,
    ):
        self.window_size = window_size
        self.mean_threshold = mean_threshold
        self.consecutive_threshold = consecutive_threshold

        self._entropies: List[float] = []
        self._consecutive_count = 0
        self._start_idx: int | None = None
        self._recent_ranges: List[Tuple[int, int]] = []
        self._detected = False

    def reset(self):
        self._entropies.clear()
        self._consecutive_count = 0
        self._start_idx = None
        self._recent_ranges.clear()
        self._detected = False

    def update(self, token_entropy: float) -> Tuple[int, float]:
        """Feed one token's entropy. Returns (detected_flag, current_mean)."""
        self._entropies.append(token_entropy)
        idx = len(self._entropies) - 1

        if idx < self.window_size - 1:
            return 0, token_entropy

        # Compute current window mean/std
        window = self._entropies[idx - self.window_size + 1: idx + 1]
        cur_mean = float(np.mean(window))
        cur_std = float(np.std(window))

        # Threshold condition
        if cur_mean > self.mean_threshold:
            self._consecutive_count = 0
            self._start_idx = None
            return 0, cur_mean

        # Stability condition — compare with recent range union
        if self._recent_ranges:
            pre_start = self._recent_ranges[0][0]
            pre_end = self._recent_ranges[-1][1]
            union_start = max(pre_start - 2, 0)
            union_end = min(pre_end, len(self._entropies) - 1)
            union_vals = self._entropies[union_start: union_end + 1]
            union_mean = float(np.mean(union_vals))
            union_std = float(np.std(union_vals))
        else:
            union_mean, union_std = cur_mean, cur_std

        esp = 3 * union_std
        cond = (union_mean - esp) <= cur_mean <= (union_mean + esp)

        if cond:
            self._consecutive_count += 1
            if self._consecutive_count == 1:
                self._start_idx = idx - self.window_size + 1

            if self._consecutive_count >= self.consecutive_threshold:
                end_idx = idx + 1
                if self._start_idx is not None:
                    self._recent_ranges.append((self._start_idx, end_idx))
                    if len(self._recent_ranges) > 3:
                        self._recent_ranges.pop(0)
                self._detected = True
                return 1, cur_mean
        else:
            self._consecutive_count = 0
            self._start_idx = None

        return 0, cur_mean

    def get_entropy_signal(self) -> EntropySignal:
        """Return current entropy signal state."""
        if not self._entropies:
            return EntropySignal(
                mean_entropy=0.0, std_entropy=0.0,
                is_suspicious=False, confidence=0.0, anomaly_score=0.0,
            )

        window = self._entropies[-self.window_size:]
        cur_mean = float(np.mean(window))
        cur_std = float(np.std(window))

        # Anomaly score: how far below threshold
        if self.mean_threshold > 0:
            anomaly = max(0.0, (self.mean_threshold - cur_mean) / self.mean_threshold)
        else:
            anomaly = 0.0

        is_suspicious = self._detected or (cur_mean <= self.mean_threshold)
        confidence = min(anomaly, 1.0)

        return EntropySignal(
            mean_entropy=cur_mean,
            std_entropy=cur_std,
            is_suspicious=is_suspicious,
            confidence=confidence,
            anomaly_score=anomaly,
        )


def compute_step_entropies(logits_list: List) -> List[float]:
    """Compute per-step Shannon entropy from raw logits tensors.

    Args:
        logits_list: List of tensors, each shape [batch_size, vocab_size]

    Returns:
        List of entropy values per generation step.
    """
    import torch
    import torch.nn.functional as F

    entropies = []
    for logits in logits_list:
        probs = F.softmax(logits[0], dim=-1)
        entropy = float(torch.distributions.Categorical(probs).entropy().item())
        entropies.append(entropy)
    return entropies

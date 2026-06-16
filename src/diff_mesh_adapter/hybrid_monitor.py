from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridMonitorDecision:
    use_full: bool
    drift: float
    age: int
    reason: str


def hybrid_proxy_intensity(strain: np.ndarray, divergence: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    strain = np.maximum(np.asarray(strain, dtype=np.float64), 0.0)
    divergence = np.maximum(np.asarray(divergence, dtype=np.float64), 0.0)
    z_strain = strain / max(float(np.quantile(strain, 0.90)), eps)
    z_divergence = divergence / max(float(np.quantile(divergence, 0.90)), eps)
    return np.sqrt(z_strain * z_strain + z_divergence * z_divergence + eps)


def hybrid_log_correction(
        full_indicator: np.ndarray,
        proxy_indicator: np.ndarray,
        *,
        eps: float = 1.0e-12,
) -> np.ndarray:
    full = np.maximum(np.asarray(full_indicator, dtype=np.float64), eps)
    proxy = np.maximum(np.asarray(proxy_indicator, dtype=np.float64), eps)
    return np.log(full) - np.log(proxy)


def corrected_hybrid_indicator(
        proxy_indicator: np.ndarray,
        log_correction: np.ndarray,
        *,
        age: int,
        decay_base: float = 0.85,
        eps: float = 1.0e-12,
) -> np.ndarray:
    proxy = np.maximum(np.asarray(proxy_indicator, dtype=np.float64), eps)
    correction = np.asarray(log_correction, dtype=np.float64)
    decay = float(decay_base) ** max(int(age), 0)
    return np.exp(np.log(proxy) + decay * correction)


def proxy_drift(
        current_proxy: np.ndarray,
        reference_proxy: np.ndarray | None,
        *,
        eps: float = 1.0e-12,
) -> float:
    if reference_proxy is None:
        return float("inf")
    current = np.maximum(np.asarray(current_proxy, dtype=np.float64), eps)
    reference = np.maximum(np.asarray(reference_proxy, dtype=np.float64), eps)
    return float(np.quantile(np.abs(np.log(current) - np.log(reference)), 0.90))


def should_compute_full_monitor(
        *,
        adaptation_index: int,
        full_every: int,
        drift: float,
        drift_threshold: float,
        has_correction: bool,
        previous_rejected: bool,
        previous_flips: bool,
        area_ratio_worsened: bool,
) -> HybridMonitorDecision:
    full_every = max(int(full_every), 1)
    if adaptation_index == 0:
        return HybridMonitorDecision(True, drift, 0, "first")
    if not has_correction:
        return HybridMonitorDecision(True, drift, 0, "missing_correction")
    if adaptation_index % full_every == 0:
        return HybridMonitorDecision(True, drift, 0, "cadence")
    if previous_rejected:
        return HybridMonitorDecision(True, drift, 0, "previous_rejected")
    if previous_flips:
        return HybridMonitorDecision(True, drift, 0, "previous_flips")
    if area_ratio_worsened:
        return HybridMonitorDecision(True, drift, 0, "area_ratio_worsened")
    if drift > drift_threshold:
        return HybridMonitorDecision(True, drift, 0, "proxy_drift")
    return HybridMonitorDecision(False, drift, adaptation_index % full_every, "proxy")

"""Baseline (health-index) detector: learns what *this* machine looks like when healthy and
flags windows that drift away from it. Needs no fault labels, so it works on any motor or fan
once you record a few seconds of healthy running."""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

# scale-free features only, so the detector is not fooled by the phone's gain or position
SCALE_FREE = ["kurtosis", "skewness", "crest", "shape", "impulse", "margin", "spectral_centroid",
              "spectral_kurtosis", "env_kurtosis"]


def select(feats: np.ndarray, names: list[str], use_bands: bool = True) -> np.ndarray:
    cols = [i for i, n in enumerate(names) if n in SCALE_FREE or (use_bands and n.startswith("band_"))]
    return feats[:, cols]


class BaselineDetector:
    """Mahalanobis distance to the healthy baseline, threshold = a high quantile of held-out
    healthy windows. Health index = distance / threshold (>1 means alarm)."""

    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile

    def fit(self, X: np.ndarray) -> "BaselineDetector":
        if len(X) < 10:
            raise ValueError("Need at least 10 healthy windows (a few seconds of recording)")
        X = np.log1p(np.abs(X)) * np.sign(X)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        fit_idx, cal_idx = idx[: len(X) // 2], idx[len(X) // 2:]
        self.mu_ = X[fit_idx].mean(axis=0)
        self.sd_ = X[fit_idx].std(axis=0) + 1e-9
        self.cov_ = LedoitWolf().fit((X[fit_idx] - self.mu_) / self.sd_)
        d_cal = self._dist(X[cal_idx])
        self.threshold_ = float(np.quantile(d_cal, self.quantile)) * 1.1  # 10% margin
        # refit the mean/cov on all healthy data, keep the calibrated threshold
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-9
        self.cov_ = LedoitWolf().fit((X - self.mu_) / self.sd_)
        return self

    def _dist(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt(self.cov_.mahalanobis((X - self.mu_) / self.sd_))

    def health_index(self, X: np.ndarray) -> np.ndarray:
        X = np.log1p(np.abs(X)) * np.sign(X)
        return self._dist(X) / self.threshold_

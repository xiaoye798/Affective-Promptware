"""按场景整组自助重采样；同一场景的情绪、条件和重复轮次保持配对。"""
from __future__ import annotations

import numpy as np

DRAWS = 4000
SEED = 20261020


def percentile_ci(scenario_values, *, scale=1.0):
    values = np.asarray(scenario_values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("每个场景必须对应一个有限数值。")
    indices = np.random.default_rng(SEED).integers(0, len(values), size=(DRAWS, len(values)))
    estimates = values[indices].mean(axis=1) * scale
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": round(float(values.mean() * scale), 12), "ci_lo": round(float(lo), 12), "ci_hi": round(float(hi), 12),
            "scenarios": len(values), "draws": DRAWS, "seed": SEED}

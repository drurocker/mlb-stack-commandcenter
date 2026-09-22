from __future__ import annotations

from dataclasses import dataclass
import math

MODEL_VERSION = "drew-mlb-proj-v1.1"
SIGNAL_THRESHOLD = 0.10

HITTER_FACTOR_WEIGHTS = {
    "team_hitting": {"floor": 0.30, "median": 0.22, "ceiling": 0.13},
    "starter": {"floor": 0.14, "median": 0.27, "ceiling": 0.22},
    "bullpen": {"floor": 0.09, "median": 0.13, "ceiling": 0.20},
    "hard_contact": {"floor": 0.09, "median": 0.13, "ceiling": 0.20},
    "lineup_position": {"floor": 0.28, "median": 0.13, "ceiling": 0.08},
    "game_environment": {"floor": 0.10, "median": 0.12, "ceiling": 0.17},
}

PITCHER_FACTOR_WEIGHTS = {
    "pitcher_quality": {"floor": 0.40, "median": 0.40, "ceiling": 0.35},
    "opponent_hitting": {"floor": 0.32, "median": 0.32, "ceiling": 0.27},
    "opponent_hard_contact": {"floor": 0.18, "median": 0.18, "ceiling": 0.25},
    "game_environment": {"floor": 0.10, "median": 0.10, "ceiling": 0.13},
}


def _validate_weights() -> None:
    for name, mapping in (("hitter", HITTER_FACTOR_WEIGHTS), ("pitcher", PITCHER_FACTOR_WEIGHTS)):
        for distribution in ("floor", "median", "ceiling"):
            total = sum(float(factors[distribution]) for factors in mapping.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"{name} {distribution} factor weights must sum to 1.0; got {total}")


_validate_weights()


@dataclass(frozen=True)
class ProjectionModelConfig:
    model_version: str = MODEL_VERSION
    floor_sensitivity: float = 0.06
    median_sensitivity: float = 0.14
    ceiling_sensitivity: float = 0.24
    signal_threshold: float = SIGNAL_THRESHOLD


DEFAULT_MODEL_CONFIG = ProjectionModelConfig()

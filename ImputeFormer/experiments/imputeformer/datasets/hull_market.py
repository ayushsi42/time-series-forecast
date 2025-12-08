from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .prototypes import PandasDataset
from .prototypes.mixin import MissingValuesMixin

LABEL_COLUMNS: Sequence[str] = (
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
)


def _default_csv_path() -> Path:
    """Locate train.csv by walking up from this file."""
    env_override = os.getenv("HULL_MARKET_CSV")
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"HULL_MARKET_CSV points to missing file: {candidate}")

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "train.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("train.csv not found. Set HULL_MARKET_CSV env var or place train.csv at repo root.")


class HullMarketDataset(PandasDataset, MissingValuesMixin):
    """Wrap the Hull Tactical Market table as a tsl dataset.

    Each feature column (D*, E*, I*, M*, P*, S*, V*) is treated as a node with
    a single channel named "value". The temporal index is derived from
    ``date_id`` by interpreting it as a day offset from 2000-01-01.
    """

    def __init__(
        self,
        csv_path: Optional[os.PathLike[str] | str] = None,
        feature_columns: Optional[Sequence[str]] = None,
        base_date: str = "2000-01-01",
        freq: str = "D",
    ) -> None:
        csv_path = Path(csv_path) if csv_path is not None else _default_csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(f"Hull market CSV not found at {csv_path}")

        raw = pd.read_csv(csv_path)
        if "date_id" not in raw.columns:
            raise ValueError("Expected a date_id column for chronological sorting.")
        raw = raw.sort_values("date_id").reset_index(drop=True)

        if feature_columns is None:
            feature_columns = [
                c for c in raw.columns if c not in LABEL_COLUMNS and c != "date_id"
            ]
        missing_requested = set(feature_columns).difference(raw.columns)
        if missing_requested:
            raise ValueError(f"Requested columns not found: {sorted(missing_requested)}")

        index = pd.to_datetime(pd.Timestamp(base_date) + pd.to_timedelta(raw["date_id"], unit="D"))
        values = raw[feature_columns].astype(np.float32)
        multi_cols = pd.MultiIndex.from_product(
            [feature_columns, ["value"]], names=["nodes", "channels"]
        )
        feature_frame = pd.DataFrame(values.to_numpy(), index=index, columns=multi_cols)

        mask = (~values.isna()).to_numpy(dtype=np.uint8).reshape(len(values), len(feature_columns), 1)

        super().__init__(
            dataframe=feature_frame,
            mask=mask,
            freq=freq,
            similarity_score='correntropy',
            temporal_aggregation="mean",
            spatial_aggregation="mean",
            default_splitting_method="temporal",
            name="HullMarket",
        )

        eval_mask = np.zeros_like(self.mask, dtype=np.uint8)
        self.set_eval_mask(eval_mask)
        self.csv_path = str(csv_path)
        self.feature_columns = tuple(feature_columns)
        self.date_ids = raw["date_id"].to_numpy()

    @property
    def description(self) -> str:
        return (
            "Hull Tactical Market dataset with per-day macro, valuation, "
            "sentiment, volatility, and technical signals."
        )
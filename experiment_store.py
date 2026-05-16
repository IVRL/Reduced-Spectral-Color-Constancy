"""
CSV-based experiment result store for CBC runs.

Defines the column schemas (SCHEMA, RADIANCE_SCHEMA), the MODEL_SETTINGS
set that identifies which settings constitute a unique model, and
CBCExperimentStore for writing and querying experiment results.
"""

from __future__ import annotations

import csv
import os
import uuid
from enum import Enum
from datetime import datetime

import pandas as pd

from experiment_settings import DimRedMethod, CameraRGBMode
import cbc_model


SCHEMA: dict[str, type] = {
    # --- Identity ---
    "run_id":                       str,
    "timestamp":                    str,
    "experiment_name":              str,
    "run_notes":                    str,
    # --- Settings ---
    "dim_red_method":               DimRedMethod,
    "camera_rgb_mode":              CameraRGBMode,
    "camera_model":                 str,
    "random_projection_seed":       int,
    "num_components":               int,
    "threshold":                    bool,
    "histogram_bin_count":          int,
    "projection_illuminant_names":  list,
    "test_illuminant_names":        list,
    "snr_db":                       float,
    # --- Metrics ---
    "accuracy":                     float,
    "cross_entropy":                float,
    "mean_ang_err":                 float,
    "median_ang_err":               float,
    "trimean_ang_err":              float,
    "best25_mean_ang_err":          float,
    "worst25_ang_err":              float,
    "mean_ang_err_wp":              float,
    "median_ang_err_wp":            float,
    "trimean_ang_err_wp":           float,
    "best25_mean_ang_err_wp":       float,
    "worst25_ang_err_wp":           float,
    # --- Model ---
    "model_path":                   str,
}

COLUMNS = list(SCHEMA.keys())

RADIANCE_SCHEMA: dict[str, type] = {
    # --- Identity ---
    "run_id":                              str,
    "timestamp":                           str,
    "experiment_name":                     str,
    "run_notes":                           str,
    # --- Settings (same as SCHEMA, minus snr_db) ---
    "dim_red_method":                      DimRedMethod,
    "camera_rgb_mode":                     CameraRGBMode,
    "camera_model":                        str,
    "random_projection_seed":              int,
    "num_components":                      int,
    "threshold":                           bool,
    "histogram_bin_count":                 int,
    "projection_illuminant_names":         list,
    "test_illuminant_names":               list,
    # --- Metrics vs continuous GT SPD ---
    "accuracy":                            float,
    "mean_ang_err":                        float,
    "median_ang_err":                      float,
    "trimean_ang_err":                     float,
    "best25_mean_ang_err":                 float,
    "worst25_ang_err":                     float,
    "mean_ang_err_wp":                     float,
    "median_ang_err_wp":                   float,
    "trimean_ang_err_wp":                  float,
    "best25_mean_ang_err_wp":              float,
    "worst25_ang_err_wp":                  float,
    # --- Metrics vs snapped GT candidate ---
    "mean_snapped_ang_err":                float,
    "median_snapped_ang_err":              float,
    "trimean_snapped_ang_err":             float,
    "best25_mean_snapped_ang_err":         float,
    "worst25_snapped_ang_err":             float,
    "mean_snapped_ang_err_wp":             float,
    "median_snapped_ang_err_wp":           float,
    "trimean_snapped_ang_err_wp":          float,
    "best25_mean_snapped_ang_err_wp":      float,
    "worst25_snapped_ang_err_wp":          float,
    # --- Model ---
    "model_path":                          str,
}

RADIANCE_COLUMNS = list(RADIANCE_SCHEMA.keys())

# Settings that define the model (projector + histograms).
# get_model() only matches on these — snr_db and metadata must not force a rebuild.
MODEL_SETTINGS = {
    "dim_red_method",
    "camera_rgb_mode",
    "camera_model",
    "random_projection_seed",
    "num_components",
    "threshold",
    "histogram_bin_count",
    "projection_illuminant_names",
    "test_illuminant_names",
}


def _serialize(val, typ: type) -> str:
    """Convert a Python value to its CSV string representation."""
    if val is None or val == "":
        return ""
    if typ == bool:
        return str(val)
    if typ == list:
        return ";".join(map(str, val)) if isinstance(val, (list, tuple)) else str(val)
    if isinstance(typ, type) and issubclass(typ, Enum):
        return val.value if isinstance(val, Enum) else str(val)
    return str(val)


class CBCExperimentStore:
    """
    Single-CSV experiment tracker — one row per run.

    Handles serialization of settings and metrics to CSV, backwards-compatible
    loading with dtype coercion, and model lookup by settings fingerprint.
    """

    def __init__(
        self,
        base_dir: str = "results",
        file_name: str = "cbc_runs.csv",
        schema: dict[str, type] | None = None,
    ) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, file_name)
        self.schema = schema if schema is not None else SCHEMA
        self.columns = list(self.schema.keys())
        if not os.path.exists(self.path):
            self._write_header()

    def add_run(
        self,
        settings: dict,
        metrics: dict,
        model_path: str | None = None,
        run_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Serialize settings + metrics into one CSV row. Returns run_id."""
        data = {
            **settings,
            **metrics,
            "run_id":     run_id or str(uuid.uuid4()),
            "timestamp":  timestamp or datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_path": model_path or "",
        }
        row = {col: _serialize(data.get(col), typ) for col, typ in self.schema.items()}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(row)
        return row["run_id"]

    def load(self) -> pd.DataFrame:
        """Load the CSV with correct dtypes. Columns missing from old files are filled with NaN."""
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.columns)

        df = pd.read_csv(self.path, dtype=str, keep_default_na=False)

        # Backwards compatibility: add any columns absent from older CSV files
        for col in self.columns:
            if col not in df.columns:
                df[col] = ""

        for col, typ in self.schema.items():
            if typ == bool:
                df[col] = df[col].map(lambda x: True if x == "True" else (False if x == "False" else pd.NA))
            elif typ == int:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif typ == float:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif isinstance(typ, type) and issubclass(typ, Enum):
                df[col] = df[col].map(lambda x, t=typ: t(x) if x else None)
            # str and list stay as strings in the dataframe

        return df

    def best_runs(self, by: str = "accuracy", n: int = 10) -> pd.DataFrame:
        """Return the top-n runs sorted by metric (descending)."""
        df = self.load()
        if by not in df.columns:
            return pd.DataFrame()
        return df.sort_values(by=by, ascending=False, na_position="last").head(n)

    def get_model(self, settings: dict, prefer: str = "latest") -> cbc_model.Model | None:
        """
        Find and load a saved model whose settings match the given dict.

        Only MODEL_SETTINGS keys are compared — snr_db and metadata fields
        are ignored so they don't force a model rebuild.

        Args:
            settings: Settings dict (as returned by validate_kaust_settings).
            prefer:   'latest' or 'earliest' — which run to load if multiple match.

        Returns:
            Loaded Model, or None if no matching saved model is found.
        """
        df = self.load()
        if df.empty:
            return None

        mask = pd.Series(True, index=df.index)

        for col, typ in self.schema.items():
            if col not in MODEL_SETTINGS:
                continue
            val = settings.get(col)
            if val is None or col not in df.columns:
                continue
            if typ == list:
                target = frozenset(map(str, val))
                mask &= df[col].apply(
                    lambda s: frozenset(s.split(";")) if s else frozenset()
                ) == target
            else:
                mask &= df[col].astype(str) == _serialize(val, typ)

        matched = df[mask].copy()
        if matched.empty:
            return None

        if "timestamp" in matched.columns:
            matched = matched.sort_values("timestamp", ascending=(prefer == "earliest"))

        for path in matched["model_path"]:
            if isinstance(path, str) and path.strip():
                return cbc_model.Model.load(path)
        return None

    def _write_header(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.columns).writeheader()

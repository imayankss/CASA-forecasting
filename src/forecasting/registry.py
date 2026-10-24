"""
registry.py — Model registry: stores fitted models, results, and metadata.
Provides experiment tracking and reproducibility.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class ModelRecord:
    """Stores everything about a single model experiment."""
    name:        str
    params:      Dict[str, Any]
    metrics:     Dict[str, float]
    forecast:    pd.Series
    ci:          Optional[pd.DataFrame]
    residuals:   np.ndarray
    fitted:      pd.Series
    train_time:  float
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())
    tags:        List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name":       self.name,
            "params":     self.params,
            "metrics":    self.metrics,
            "train_time": self.train_time,
            "timestamp":  self.timestamp,
            "tags":       self.tags,
        }


class ModelRegistry:
    """
    Central registry for trained model records.

    Usage
    -----
    registry = ModelRegistry()
    registry.register(record)
    best = registry.best_by("MAPE")
    registry.save_index("reports/registry.json")
    """

    def __init__(self) -> None:
        self._records: Dict[str, ModelRecord] = {}

    def register(self, record: ModelRecord) -> None:
        self._records[record.name] = record

    def get(self, name: str) -> Optional[ModelRecord]:
        return self._records.get(name)

    def all_names(self) -> List[str]:
        return list(self._records.keys())

    def all_records(self) -> Dict[str, ModelRecord]:
        return dict(self._records)

    def metrics_df(self) -> pd.DataFrame:
        rows = {name: rec.metrics for name, rec in self._records.items()}
        return pd.DataFrame(rows).T.round(4)

    def best_by(self, metric: str = "MAPE", lower_is_better: bool = True) -> Optional[str]:
        if not self._records:
            return None
        fn = min if lower_is_better else max
        return fn(
            self._records,
            key=lambda k: self._records[k].metrics.get(metric, float("inf")),
        )

    def save_index(self, path: str) -> None:
        """Save a JSON index of all experiments."""
        index = {name: rec.to_dict() for name, rec in self._records.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(index, f, indent=2, default=str)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for name, rec in self._records.items():
            row = {"Model": name, "Train_Time_s": round(rec.train_time, 2)}
            row.update({k: round(v, 4) for k, v in rec.metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("Model")


def run_metadata(config: Dict, data_files: List[str]) -> Dict:
    """Provenance for one run: config, input file sizes and hashes, git commit."""
    import hashlib
    import subprocess
    from datetime import datetime
    from pathlib import Path

    def sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip()

    try:
        git_sha = git("rev-parse", "HEAD")
        # Uncommitted tracked changes mean git_sha is not the code that produced this run
        git_dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_dirty = None, None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config": config,
        "data_snapshot": [
            {"path": p, "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in data_files
        ],
    }

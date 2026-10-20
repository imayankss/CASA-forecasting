"""
test_registry_metadata.py — run_metadata records config, data hash and git SHA.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.registry import run_metadata


def test_round_trip(tmp_path):
    data = tmp_path / "input.csv"
    data.write_text("a,b\n1,2\n")
    meta = run_metadata({"h": 4}, [str(data)])
    back = json.loads(json.dumps(meta, default=str))
    assert back["config"] == {"h": 4}
    snap = back["data_snapshot"][0]
    assert snap["bytes"] == data.stat().st_size
    assert snap["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert "git_sha" in back and "timestamp" in back

#!/usr/bin/env python
"""Validate a checkpoint .json + .npz pair.

Usage:
    uv run python scripts/validate_checkpoint.py path/to/checkpoint_prefix
    (expects checkpoint_prefix.json and checkpoint_prefix.npz)
"""

import json
import sys
from pathlib import Path

import numpy as np
import jsonschema


def validate(prefix: str) -> bool:
    json_path = Path(prefix).with_suffix(".json")
    npz_path = Path(prefix).with_suffix(".npz")

    if not json_path.exists():
        print(f"MISSING: {json_path}")
        return False
    if not npz_path.exists():
        print(f"MISSING: {npz_path}")
        return False

    schema_path = Path(__file__).parent.parent / "schemas" / "checkpoint.schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    with open(json_path) as f:
        meta = json.load(f)

    try:
        jsonschema.validate(meta, schema)
    except jsonschema.ValidationError as e:
        print(f"INVALID JSON: {json_path}")
        print(f"  Error: {e.message}")
        return False

    # Check npz has expected keys.
    data = dict(np.load(npz_path, allow_pickle=False))
    expected_keys = ["archive/thresholds", "archive/occupied_indices"]
    missing = [k for k in expected_keys if k not in data]
    if missing:
        print(f"INVALID NPZ: missing keys {missing}")
        return False

    gen = meta.get("generation", "?")
    n_occupied = meta.get("archive", {}).get("occupied_count", "?")
    print(f"VALID: {prefix}")
    print(f"  generation: {gen}")
    print(f"  archive: {n_occupied} occupied cells")
    print(f"  npz keys: {len(data)}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_checkpoint.py <checkpoint_prefix>")
        sys.exit(1)
    ok = all(validate(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)

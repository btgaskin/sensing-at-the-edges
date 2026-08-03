#!/usr/bin/env python
"""Validate a manifest.json against the JSON Schema.

Usage:
    uv run python scripts/validate_manifest.py path/to/manifest.json
    uv run python scripts/validate_manifest.py outputs/smoke/manifest.json
"""

import json
import sys
from pathlib import Path

import jsonschema


def validate(manifest_path: str) -> bool:
    schema_path = Path(__file__).parent.parent / "schemas" / "manifest.schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    try:
        jsonschema.validate(manifest, schema)
        print(f"VALID: {manifest_path}")
        print(f"  run_id: {manifest.get('run_id', '?')}")
        print(f"  status: {manifest.get('status', '?')}")
        print(f"  preset: {manifest.get('preset_id', '?')}")
        return True
    except jsonschema.ValidationError as e:
        print(f"INVALID: {manifest_path}")
        print(f"  Error: {e.message}")
        print(f"  Path: {list(e.absolute_path)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_manifest.py <manifest.json>")
        sys.exit(1)
    ok = all(validate(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)

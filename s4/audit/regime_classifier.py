#!/usr/bin/env python3
"""Workstream-B regime classifier for the S4 validity audit (read-only analysis).

Compares two `per_episode.csv` post-run exports (e.g. local-current vs updated-local, or
either-vs-archived) and answers the two questions the audit hinges on:

  1. Mismatch taxonomy: per (assay, deformation_condition, distance_pair/width, side), classify
     the A-vs-B difference as one of:
        exact            - success/fell agree and final_com_x within --com-tol
        step-budget      - surface_max_steps differs (difference may be horizon, not regime)
        regime-sensitive - success or fell FLIPS with the same task matrix + same horizon
        unresolved       - missing rows / NaNs / can't classify

  2. Structured-vs-artifact (the decisive S4 question): does SAFETY stay robust while only
     REVERSAL shifts? Reports, per regime, the fall rate (safety) and the reversal signature
     (turn_count, initial_direction, success). A clean "fall-rate stable, reversal-timing/
     direction shifts" pattern SUPPORTS the knife-edge reading; success/fall incoherence KILLS it.

This does NOT run any simulation. It only reads CSVs already produced by `post_run`. Run it when
the author's reruns land. See s4/audit/AUDIT_REPORT.md and CLAIMS_LEDGER.md.

Usage:
    python regime_classifier.py \
        --a /path/to/local_current/.../per_episode.csv  --label-a local_current \
        --b /path/to/updated_local/.../per_episode.csv   --label-b updated_local \
        --com-tol 0.05 --out report

Writes <out>.md and <out>.json next to --out (default: ./regime_compare).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Episode key: identifies a comparable task slot across two runs. surface_max_steps is kept
# OUT of the key so a horizon change shows up as a step-budget difference, not a missing row.
KEY = ["assay_family", "deformation_condition", "signaling_condition", "distance_pair", "side", "width", "surface_id"]
# Outcome buckets used for the structured-vs-artifact view.
OUTCOMES = ["fall", "full_reverse_and_goal", "direct_to_goal", "cliff_stop_only", "other"]


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in KEY + ["success", "fell", "survived", "turn_count", "initial_direction",
                       "final_com_x", "surface_max_steps", "deathplane_contact", "left_exit"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("success", "fell", "survived", "deathplane_contact", "left_exit"):
        df[col] = df[col].map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"} if pd.notna(v) else np.nan)
    df["turn_count"] = pd.to_numeric(df["turn_count"], errors="coerce")
    df["final_com_x"] = pd.to_numeric(df["final_com_x"], errors="coerce")
    df["surface_max_steps"] = pd.to_numeric(df["surface_max_steps"], errors="coerce")
    return df


def classify_outcome(row: pd.Series) -> str:
    """Bucket one episode. Fall dominates; reversal = reached goal after >=1 turn."""
    if row.get("fell") is True or row.get("deathplane_contact") is True or row.get("left_exit") is True:
        return "fall"
    success = row.get("success") is True
    turned = (row.get("turn_count") or 0) >= 1
    if success and turned:
        return "full_reverse_and_goal"
    if success and not turned:
        return "direct_to_goal"
    if not success:
        return "cliff_stop_only"  # safe miss: didn't fall, didn't reach goal
    return "other"


def _agg_key(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse trials to one row per task slot (means over trials)."""
    df = df.copy()
    df["outcome"] = df.apply(classify_outcome, axis=1)
    grp = df.groupby([c for c in KEY], dropna=False)
    out = grp.agg(
        n=("success", "size"),
        success_rate=("success", "mean"),
        fall_rate=("fell", "mean"),
        mean_turn_count=("turn_count", "mean"),
        mean_final_com_x=("final_com_x", "mean"),
        surface_max_steps=("surface_max_steps", "max"),
    ).reset_index()
    # majority initial_direction + majority outcome per slot
    out["initial_direction"] = grp["initial_direction"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "").values
    out["outcome"] = grp["outcome"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else "other").values
    return out


def classify_mismatch(a: pd.Series, b: pd.Series, com_tol: float) -> str:
    # KNOWN LIMITATION: double_cliff episodes score with `survived`/`width` rather than
    # `success`/`final_com_x`, so their slots land in `unresolved` here. This classifier targets
    # the native_core / long_distance deformation comparison (where `success` is populated); for
    # double_cliff use the `survived`-based survival-by-width charts in post_run instead.
    if a is None or b is None:
        return "unresolved"
    sr_a, sr_b = a["success_rate"], b["success_rate"]
    fr_a, fr_b = a["fall_rate"], b["fall_rate"]
    if any(pd.isna(x) for x in (sr_a, sr_b, fr_a, fr_b)):
        return "unresolved"
    com_close = abs((a["mean_final_com_x"] or 0) - (b["mean_final_com_x"] or 0)) <= com_tol
    outcome_same = a["outcome"] == b["outcome"]
    if outcome_same and com_close:
        return "exact"
    if not np.isclose(a["surface_max_steps"], b["surface_max_steps"], equal_nan=True):
        return "step-budget"
    # same horizon, same task slot, but success/fall or outcome flips
    if (round(sr_a, 6) != round(sr_b, 6)) or (round(fr_a, 6) != round(fr_b, 6)) or not outcome_same:
        return "regime-sensitive"
    return "unresolved"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="per_episode.csv for regime A")
    ap.add_argument("--b", required=True, help="per_episode.csv for regime B")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--com-tol", type=float, default=0.05, help="final_com_x tolerance for 'exact' (world units)")
    ap.add_argument("--out", default="regime_compare", help="output stem (.md + .json)")
    args = ap.parse_args()

    A, B = _agg_key(_read(args.a)), _agg_key(_read(args.b))
    merged = A.merge(B, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)

    rows, tax = [], {"exact": 0, "step-budget": 0, "regime-sensitive": 0, "unresolved": 0}
    for _, m in merged.iterrows():
        a = m.filter(regex="_a$").rename(lambda c: c[:-2]) if m["_merge"] != "right_only" else None
        b = m.filter(regex="_b$").rename(lambda c: c[:-2]) if m["_merge"] != "left_only" else None
        verdict = classify_mismatch(a, b, args.com_tol)
        tax[verdict] += 1
        rows.append({**{k: m[k] for k in KEY}, "verdict": verdict,
                     "success_a": None if a is None else a.get("success_rate"),
                     "success_b": None if b is None else b.get("success_rate"),
                     "fall_a": None if a is None else a.get("fall_rate"),
                     "fall_b": None if b is None else b.get("fall_rate"),
                     "outcome_a": None if a is None else a.get("outcome"),
                     "outcome_b": None if b is None else b.get("outcome")})

    # Structured-vs-artifact: dissociation between safety (fall) and reversal (turn/success).
    def _means(df, mask=None):
        d = df if mask is None else df[mask]
        return {"fall_rate": float(np.nanmean(d["fall_rate"])) if len(d) else float("nan"),
                "success_rate": float(np.nanmean(d["success_rate"])) if len(d) else float("nan"),
                "mean_turn_count": float(np.nanmean(d["mean_turn_count"])) if len(d) else float("nan")}
    dissoc = {args.label_a: _means(A), args.label_b: _means(B)}
    safety_delta = abs(dissoc[args.label_a]["fall_rate"] - dissoc[args.label_b]["fall_rate"])
    reversal_delta = abs(dissoc[args.label_a]["mean_turn_count"] - dissoc[args.label_b]["mean_turn_count"])
    structured = bool(safety_delta < 0.1 and reversal_delta >= 0.1)  # safety stable, reversal shifts

    report = {"a": args.label_a, "b": args.label_b, "com_tol": args.com_tol,
              "taxonomy": tax, "n_slots": len(rows), "dissociation": dissoc,
              "safety_delta": safety_delta, "reversal_delta": reversal_delta,
              "structured_pattern": structured, "slots": rows}
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))

    lines = [f"# Regime comparison: {args.label_a} vs {args.label_b}", "",
             f"Slots compared: {len(rows)}  |  com_tol={args.com_tol}", "",
             "## Mismatch taxonomy", ""]
    for k in ("exact", "step-budget", "regime-sensitive", "unresolved"):
        lines.append(f"- **{k}**: {tax[k]}")
    lines += ["", "## Structured-vs-artifact (safety vs reversal)", "",
              f"- {args.label_a}: fall={dissoc[args.label_a]['fall_rate']:.3f}  "
              f"success={dissoc[args.label_a]['success_rate']:.3f}  turn={dissoc[args.label_a]['mean_turn_count']:.3f}",
              f"- {args.label_b}: fall={dissoc[args.label_b]['fall_rate']:.3f}  "
              f"success={dissoc[args.label_b]['success_rate']:.3f}  turn={dissoc[args.label_b]['mean_turn_count']:.3f}",
              f"- safety Δ (fall-rate) = {safety_delta:.3f}; reversal Δ (turn-count) = {reversal_delta:.3f}",
              "",
              "**Verdict:** " + ("STRUCTURED — safety robust, reversal shifts (supports knife-edge)."
                                 if structured else
                                 "NOT a clean dissociation — inspect regime-sensitive slots below "
                                 "(safety also moved, or reversal stable). Could indicate artifact."),
              "", "## Regime-sensitive slots (same task + horizon, outcome flips)", ""]
    for r in rows:
        if r["verdict"] == "regime-sensitive":
            lines.append(f"- {r['assay_family']}/{r['deformation_condition']}/{r['distance_pair']}/{r['side']}: "
                         f"success {r['success_a']}→{r['success_b']}, fall {r['fall_a']}→{r['fall_b']}, "
                         f"outcome {r['outcome_a']}→{r['outcome_b']}")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    print(f"Taxonomy: {tax}  | structured_pattern={structured}")


if __name__ == "__main__":
    main()

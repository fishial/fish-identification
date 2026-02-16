#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple post-training evaluation results.

Inputs: one or more paths pointing to either:
  - a directory containing metrics.json
  - a direct path to metrics.json

Outputs:
  - prints a sorted table to stdout
  - optionally writes a CSV summary
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Ensure project root (containing "fish-identification") is on sys.path
CURRENT_FILE = os.path.abspath(__file__)
DELIMITER = "fish-identification"
pos = CURRENT_FILE.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FILE[: pos + len(DELIMITER)])


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_metrics_json(p: str) -> str:
    if os.path.isdir(p):
        candidate = os.path.join(p, "metrics.json")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"No metrics.json in directory: {p}")
    if os.path.isfile(p) and p.endswith(".json"):
        return p
    raise FileNotFoundError(f"Path is neither a metrics.json nor a directory containing it: {p}")


def _flatten_row(d: dict) -> Dict[str, Any]:
    m = d.get("metrics", {})
    return {
        "path": d.get("_path", ""),
        "checkpoint": d.get("checkpoint", ""),
        "dataset_name": d.get("dataset_name", ""),
        "backbone_model_name": d.get("backbone_model_name", ""),
        "image_size": d.get("image_size", ""),
        "embedding_dim": d.get("embedding_dim", ""),
        "arcface_s": d.get("arcface_s", ""),
        "arcface_m": d.get("arcface_m", ""),
        "n_samples": m.get("n_samples", ""),
        "accuracy_top1": m.get("accuracy_top1", ""),
        "accuracy_top5": m.get("accuracy_top5", ""),
        "accuracy_top10": m.get("accuracy_top10", ""),
        "accuracy_macro_top1": m.get("accuracy_macro_top1", ""),
    }


def _print_table(rows: List[dict], columns: List[str]) -> None:
    # minimal pretty print without external deps
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def _write_csv(path: str, rows: List[dict], columns: List[str]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def get_args():
    p = argparse.ArgumentParser(description="Compare post-training eval results (metrics.json).")
    p.add_argument("paths", nargs="+", help="Paths to eval dirs or metrics.json files.")
    p.add_argument("--sort_by", type=str, default="accuracy_top5", help="Metric key to sort by.")
    p.add_argument("--descending", action="store_true", default=True, help="Sort descending (default).")
    p.add_argument("--csv_out", type=str, default=None, help="Optional path to write CSV summary.")
    return p.parse_args()


def main():
    args = get_args()
    metric_paths = [_find_metrics_json(p) for p in args.paths]
    raw = []
    for mp in metric_paths:
        d = _load_json(mp)
        d["_path"] = os.path.abspath(os.path.dirname(mp))
        raw.append(d)

    rows = [_flatten_row(d) for d in raw]
    sort_key = args.sort_by

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return float("-inf")

    rows.sort(key=lambda r: _to_float(r.get(sort_key, "")), reverse=bool(args.descending))

    cols = [
        "path",
        "backbone_model_name",
        "image_size",
        "embedding_dim",
        "accuracy_top1",
        "accuracy_top5",
        "accuracy_top10",
        "accuracy_macro_top1",
        "n_samples",
    ]
    _print_table(rows, cols)
    if args.csv_out:
        _write_csv(args.csv_out, rows, cols)
        print(f"\n[compare] Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()


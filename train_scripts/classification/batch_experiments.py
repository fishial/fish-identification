#!/usr/bin/env python3
"""
Utility to schedule multiple `lightning_train.py` experiments in sequence.

Each entry takes a JSON object (or YAML if you provide `.yaml/.yml`).
You can pass `"dataset_name"` (required), plus any CLI overrides.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def load_experiments(cfg_path: Path) -> List[Dict]:
    text = cfg_path.read_text()
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        base = yaml.safe_load(text)
    else:
        base = json.loads(text)
    if not isinstance(base, list):
        raise ValueError("Config must be a list of experiment objects.")
    if not all(isinstance(entry, dict) for entry in base):
        raise ValueError("Each experiment entry must be an object.")

    def expand_entry(entry: Dict) -> List[Dict]:
        grid_items = [(k, v) for k, v in entry.items() if isinstance(v, list)]
        if not grid_items:
            return [entry]
        key, values = grid_items[0]
        rest = dict(entry)
        rest.pop(key)
        expanded = []
        for val in values:
            new_entry = dict(rest)
            new_entry[key] = val
            expanded += expand_entry(new_entry)
        return expanded

    experiments = []
    for entry in base:
        experiments += expand_entry(entry)
    return experiments


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = Path(f"{path}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def build_cmd(entry: Dict, output_root: Path, train_script: Path) -> Tuple[List[str], Path]:
    if "dataset_name" not in entry:
        raise ValueError("Each experiment must define dataset_name.")
    cmd = [sys.executable, str(train_script)]
    cmd += ["--dataset_name", str(entry["dataset_name"])]
    if "output_dir" in entry:
        base_out = Path(entry["output_dir"])
    else:
        base_out = output_root
    run_name = entry.get("name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    run_dir = ensure_unique_dir(base_out / run_name)
    cmd += ["--output_dir", str(run_dir)]
    for key, value in entry.items():
        if key in {"dataset_name", "name", "output_dir", "note"}:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, (int, float, str)):
            cmd.extend([flag, str(value)])
        else:
            raise ValueError(f"Unsupported value for {key}: {value}")
    return cmd, run_dir


def run_experiment(cmd: List[str], log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            f.write(line)
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Experiment failed (see {log_file})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple lightning_train.py experiments sequentially."
    )
    parser.add_argument("config", type=Path, help="Path to JSON/YAML config")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between runs (seconds)")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Base output dir (default: /home/fishial/Fishial/Experiments/v10/batch_runs/<timestamp>)",
    )
    parser.add_argument(
        "--train_script",
        type=Path,
        default=Path(__file__).resolve().parent / "lightning_train.py",
        help="Path to lightning_train.py",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the experiment list continuously",
    )
    parser.add_argument(
        "--max_days",
        type=float,
        default=0.0,
        help="Stop after N days (requires --loop or N>0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path("/home/fishial/Fishial/Experiments/v10/batch_runs")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    train_script = args.train_script
    experiments = load_experiments(args.config)
    loop_enabled = args.loop or args.max_days > 0
    end_time = start_time + (args.max_days * 86400.0) if args.max_days > 0 else None

    cycle = 0
    while True:
        cycle += 1
        for entry in experiments:
            cmd, run_dir = build_cmd(entry, output_root, train_script)
            print(f"\n==== Running {entry.get('name', run_dir.name)} (cycle {cycle}) ====")
            run_experiment(cmd, run_dir)
            if args.delay > 0:
                print(f"Sleeping {args.delay}s between experiments")
                time.sleep(args.delay)
        if not loop_enabled:
            break
        if end_time is not None and time.time() >= end_time:
            print("Reached max_days limit; stopping.")
            break


if __name__ == "__main__":
    main()

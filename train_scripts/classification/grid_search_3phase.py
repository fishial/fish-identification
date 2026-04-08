#!/usr/bin/env python3
"""
3-Phase Grid Search for Fish Classification (755 classes, strong imbalance).

Phase 1: Train on classes with >=150 images (394 classes) — learn strong base embeddings.
Phase 2: Fine-tune on all 755 classes, capped at 150 images/class — add rare classes.
Phase 3: Boost large classes by restoring full data (up to 2000 images/class).

Usage:
    python grid_search_3phase.py --mode full          # run all grid combinations
    python grid_search_3phase.py --mode random --n 8  # random sample of 8 configs
    python grid_search_3phase.py --mode resume         # resume from last completed phase
    python grid_search_3phase.py --dry_run             # print commands without executing
"""

import argparse
import copy
import glob
import itertools
import json
import logging
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRAIN_SCRIPT = Path(__file__).resolve().parent / "lightning_train.py"
# Matches filenames like: model-epoch46-acc8696.ckpt  OR  model-epoch05-acc86.96.ckpt
ACC_PATTERN = re.compile(r"acc(?P<acc>\d+(?:[\._]\d+)?)")  # handles both int and float forms

DATASET_NAME = "segmentation_dataset_v0.10_with_meta"
BASE_OUTPUT_DIR = "/home/andrew/Andrew/Fishial2402/Experiments/v11"
CLASS_MAPPING = "/home/andrew/Andrew/Fishial2402/fish-identification/class_mapping.json"
LABELS_ALL = "/home/andrew/Andrew/Fishial2402/fish-identification/labels.txt"
LABELS_150 = "/home/andrew/Andrew/Fishial2402/fish-identification/labels_150.txt"
MAX_EPOCHS = 50
PHASE_TIMEOUT_SECONDS = 48 * 3600  # 48h safety limit per phase
MIN_CKPT_SIZE_BYTES = 10 * 1024    # 10 KB — smaller means corrupted
MIN_DISK_GB = 5                     # warn if less than 5 GB free
# Poll interval when checking subprocess liveness during timeout guard
_PROC_POLL_INTERVAL = 30           # seconds

_SHUTDOWN_REQUESTED = False

# ---------------------------------------------------------------------------
# Logging setup — writes to both console and a grid_search.log in output_root
# ---------------------------------------------------------------------------
_log = logging.getLogger("grid_search")


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "grid_search.log")
    fmt = "%(asctime)s %(levelname)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Search space — edit the lists to add / remove candidates
# ---------------------------------------------------------------------------
SEARCH_SPACE: Dict[str, List[Any]] = {
    "backbone_model_name": ["vit_base_patch14_reg4_dinov2.lvd142m", "beitv2_base_patch16_224.in1k_ft_in22k_in1k"],
    "embedding_dim": [512],
    "pooling_type": ["attention"],
    "neck_type": ["bnneck", "lnneck", "simple", "mlp"], # <--- ДОБАВИЛИ SIMPLE
    "use_cls_token": [True],
    "head_type": ["subcenter"],
    "arcface_s": [30.0, 64.0],
    "arcface_m": [0.2, 0.35],
    "arcface_K": [1, 3],
    "num_attention_heads": [1, 2, 3],
    "metric_loss_type": ["circle"],
    "label_smoothing": [0.05, 0.1],
}

FIXED_PARAMS: Dict[str, Any] = {
    "dataset_name": DATASET_NAME,
    "train_tag": "train",
    "val_tag": "val",
    "image_size": 224,
    "backbone_img_size": 224,
    "loss_type": "combined",
    "miner_type": "multi_similarity",
    "arcface_weight": 0.8,
    "metric_weight": 0.2,
    "use_cross_batch_memory": True,
    "memory_size": 4096,
    "augmentation_preset": "strong",
    "classes_per_batch": 48,
    "samples_per_class": 4,
    "accumulate_grad_batches": 2,
    "weight_decay": 0.1,
    "gradient_clip_val": 1.0,
    "embedding_dropout_rate": 0.3,
    "bbox_padding_limit": 0.15,
    "bg_removal_prob": 0.3,
    "class_mapping_path": CLASS_MAPPING,
}

PHASE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "phase1": {
        "labels_path": LABELS_150,
        "min_samples_per_class": 150,
        "max_samples_per_class": 2000,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": 1e-4,
        "lr_eta_min": 1e-6,
        "freeze_backbone_epochs": 2,
        "attention_warmup_epochs": 2,
        "attention_loss_lambda": 0.5,
        "use_swa": True,
        "swa_lrs": 1e-6,
        "swa_epoch_start": 0.8,
    },
    "phase2": {
        "labels_path": LABELS_ALL,
        "min_samples_per_class": 1,
        "max_samples_per_class": 150,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": 3e-5,
        "lr_eta_min": 1e-7,
        "freeze_backbone_epochs": 0,
        "attention_warmup_epochs": 1,
        "attention_loss_lambda": 0.3,
        "arcface_weight": 0.4,
        "metric_weight": 0.6,
        "use_swa": True,
        "swa_lrs": 5e-7,
        "swa_epoch_start": 0.75,
    },
    "phase3": {
        "labels_path": LABELS_ALL,
        "min_samples_per_class": 1,
        "max_samples_per_class": 2000,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": 1e-5,
        "lr_eta_min": 1e-7,
        "freeze_backbone_epochs": 30,
        "attention_warmup_epochs": 0,
        "attention_loss_lambda": 0.0,
        "arcface_weight": 1.0,
        "metric_weight": 0.0,
        "use_swa": True,
        "swa_lrs": 3e-7,
        "swa_epoch_start": 0.7,
        "augmentation_preset": "medium",
    },
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_phase_weights(phase: str, params: Dict[str, Any]) -> None:
    """Warn if arcface_weight + metric_weight don't sum to ~1.0."""
    aw = params.get("arcface_weight", 0.0)
    mw = params.get("metric_weight", 0.0)
    total = aw + mw
    if abs(total - 1.0) > 0.05:
        _log.warning(
            "%s: arcface_weight(%.2f) + metric_weight(%.2f) = %.2f (expected ~1.0)",
            phase, aw, mw, total,
        )


# ---------------------------------------------------------------------------
# ExperimentState dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentState:
    """Tracks per-experiment progress across phases."""
    name: str
    grid_params: Dict[str, Any]
    phase_dirs: Dict[str, str] = field(default_factory=dict)
    phase_checkpoints: Dict[str, str] = field(default_factory=dict)
    phase_status: Dict[str, str] = field(default_factory=dict)  # pending | running | done | failed
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentState":
        # Forward-compatible: ignore unknown keys, fill missing with defaults
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(phase_dir: str) -> Optional[str]:
    """
    Find the best checkpoint in a completed phase directory.

    Searches RECURSIVELY because lightning_train.py creates a subdirectory:
        <phase_dir>/<backbone>_<timestamp>/checkpoints/*.ckpt

    Priority:
      1. Checkpoint whose filename contains the highest acc score.
      2. Any epoch checkpoint sorted by mtime (newest first).
      3. last.ckpt as a last resort.
    """
    if not os.path.isdir(phase_dir):
        return None

    # Recursive glob: finds checkpoints at any depth inside phase_dir
    all_ckpts = glob.glob(os.path.join(phase_dir, "**", "*.ckpt"), recursive=True)
    if not all_ckpts:
        _log.debug("find_best_checkpoint: no .ckpt files found under %s", phase_dir)
        return None

    # Separate last.ckpt so it gets lowest priority
    epoch_ckpts = [c for c in all_ckpts if os.path.basename(c) != "last.ckpt"]
    last_ckpts = [c for c in all_ckpts if os.path.basename(c) == "last.ckpt"]

    best_score = float("-inf")
    best_ckpt: Optional[str] = None
    for candidate in epoch_ckpts:
        match = ACC_PATTERN.search(os.path.basename(candidate))
        if match:
            # acc value may be stored as integer (e.g. acc8696 = 86.96%) or float
            raw = match.group("acc").replace("_", ".")
            score = float(raw)
            if score > best_score:
                best_score = score
                best_ckpt = candidate

    if best_ckpt:
        _log.debug("find_best_checkpoint: best by acc=%.2f → %s", best_score, best_ckpt)
        return best_ckpt

    # No acc-scored checkpoints — pick newest epoch ckpt
    if epoch_ckpts:
        newest = max(epoch_ckpts, key=os.path.getmtime)
        _log.debug("find_best_checkpoint: best by mtime → %s", newest)
        return newest

    # Absolute fallback: last.ckpt (pick newest if multiple)
    if last_ckpts:
        newest_last = max(last_ckpts, key=os.path.getmtime)
        _log.debug("find_best_checkpoint: fallback last.ckpt → %s", newest_last)
        return newest_last

    return None


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_grid_combinations(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def sample_random_configs(search_space: Dict[str, List[Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    configs: List[Dict[str, Any]] = []
    keys = sorted(search_space.keys())
    seen: set = set()
    max_unique = 1
    for v in search_space.values():
        max_unique *= len(v)
    attempts = min(n * 20, max_unique * 3)
    for _ in range(attempts):
        combo = tuple(rng.choice(search_space[k]) for k in keys)
        if combo not in seen:
            seen.add(combo)
            configs.append(dict(zip(keys, combo)))
        if len(configs) >= n:
            break
    if len(configs) < n:
        _log.warning(
            "Requested %d random configs but only %d unique combinations exist.", n, len(configs)
        )
    return configs


# ---------------------------------------------------------------------------
# Experiment naming
# ---------------------------------------------------------------------------

def make_experiment_name(grid_params: Dict[str, Any], idx: int) -> str:
    parts = [f"exp{idx:03d}"]
    abbrevs = {
        "embedding_dim": "dim",
        "pooling_type": "pool",
        "neck_type": "neck",
        "use_cls_token": "cls",
        "arcface_s": "s",
        "arcface_m": "m",
        "arcface_K": "K",
        "num_attention_heads": "heads",
        "metric_loss_type": "mloss",
        "label_smoothing": "ls",
        "head_type": "head",
        "use_dynamic_margin": "dynm",
    }
    # Include a short backbone tag so names are unique even when backbone varies
    backbone = grid_params.get("backbone_model_name", "")
    if backbone:
        # e.g. "beitv2_base..." → "beit", "vit_base..." → "vit"
        backbone_tag = backbone.split("_")[0][:6]
        parts.append(f"bb{backbone_tag}")

    for key in sorted(grid_params.keys()):
        if key == "backbone_model_name":
            continue
        short = abbrevs.get(key, key[:4])
        val = grid_params[key]
        if isinstance(val, bool):
            val = "T" if val else "F"
        elif isinstance(val, float):
            val = f"{val:g}"
        parts.append(f"{short}{val}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def build_phase_cmd(
    grid_params: Dict[str, Any],
    phase: str,
    output_dir: str,
    prev_checkpoint: Optional[str] = None,
) -> List[str]:
    params: Dict[str, Any] = {}
    params.update(FIXED_PARAMS)
    params.update(grid_params)
    params.update(PHASE_OVERRIDES[phase])
    params["output_dir"] = output_dir

    if prev_checkpoint:
        params["load_weights_from_checkpoint"] = prev_checkpoint

    # Remove attention-specific args when not using attention pooling
    if params.get("pooling_type") != "attention":
        params.pop("num_attention_heads", None)
        params.pop("use_cls_token", None)

    # Validate weight sum before building the command
    _validate_phase_weights(phase, params)

    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    for key, val in sorted(params.items()):
        flag = f"--{key}"
        if isinstance(val, bool):
            cmd.extend([flag, "true" if val else "false"])
        else:
            cmd.extend([flag, str(val)])
    return cmd


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------

def run_phase(
    cmd: List[str],
    log_dir: str,
    phase: str,
    timeout: int = PHASE_TIMEOUT_SECONDS,
) -> int:
    """
    Launch a training subprocess, stream its output, and enforce a wall-clock timeout.

    The timeout is checked by polling the process every _PROC_POLL_INTERVAL seconds
    so it fires even if the subprocess produces no output (e.g. deadlock / GPU hang).

    Returns the process exit code, or:
      -9  if the timeout was exceeded
      -1  if an unexpected Python exception occurred
    """
    os.makedirs(log_dir, exist_ok=True)
    log_name = phase.replace("/", "_")
    log_file = os.path.join(log_dir, f"{log_name}.log")
    t0 = time.monotonic()

    _log.info("=" * 70)
    _log.info("  PHASE: %s  |  Log: %s", phase, log_file)
    _log.info("=" * 70)
    _log.info("CMD: %s ... (%d args total)", " ".join(cmd[:6]), len(cmd))
    _log.info("Full command: %s", " ".join(cmd))

    proc: Optional[subprocess.Popen] = None
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# {phase} started at {datetime.now().isoformat()}\n")
            f.write(" ".join(cmd) + "\n\n")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None

            def _kill_proc(reason: str) -> None:
                _log.warning("  Terminating process: %s", reason)
                proc.terminate()  # type: ignore[union-attr]
                try:
                    proc.wait(timeout=30)  # type: ignore[union-attr]
                except subprocess.TimeoutExpired:
                    _log.warning("  Process did not terminate in 30s — sending SIGKILL")
                    proc.kill()  # type: ignore[union-attr]
                    proc.wait()  # type: ignore[union-attr]
                f.write(f"\n# KILLED: {reason}\n")

            # Read stdout in a non-blocking-friendly way: we read line by line
            # but also periodically check wall-clock and shutdown flag.
            import select as _select

            fd = proc.stdout.fileno()

            while True:
                # Check if there's output available (100 ms window)
                ready, _, _ = _select.select([fd], [], [], _PROC_POLL_INTERVAL)

                elapsed = time.monotonic() - t0

                if timeout and elapsed > timeout:
                    _kill_proc(f"exceeded {timeout}s timeout")
                    _log.error("  TIMEOUT: phase %s killed after %.0fs", phase, elapsed)
                    return -9

                if _SHUTDOWN_REQUESTED:
                    _kill_proc("shutdown requested by signal")
                    return -9

                if ready:
                    line = proc.stdout.readline()
                    if line:
                        f.write(line)
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    else:
                        # EOF — process finished
                        break
                else:
                    # Timeout on select — process may still be running, loop again
                    if proc.poll() is not None:
                        # Process exited silently
                        break

            proc.wait()
            elapsed = time.monotonic() - t0
            f.write(f"\n# exit_code={proc.returncode}  elapsed={elapsed:.0f}s\n")

        _log.info("  Phase finished in %.0fs (exit=%s)", elapsed, proc.returncode)
        return proc.returncode

    except Exception as e:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        _log.exception("  EXCEPTION in run_phase(%s): %s", phase, e)
        return -1


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(state_file: str) -> Dict[str, "ExperimentState"]:
    """Load state with fallback to backup if main file is corrupted."""
    for path in [state_file, state_file + ".bak"]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return {name: ExperimentState.from_dict(d) for name, d in data.items()}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            _log.warning("Failed to load state from %s: %s", path, e)
    return {}


def save_state(state_file: str, states: Dict[str, "ExperimentState"]) -> None:
    """Atomic save: write to temp file, then rename (prevents corruption on crash)."""
    payload = json.dumps({name: s.to_dict() for name, s in states.items()}, indent=2)
    dir_name = os.path.dirname(state_file) or "."
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix=".state_")
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(state_file):
            shutil.copy2(state_file, state_file + ".bak")
        os.replace(tmp_path, state_file)
    except OSError as e:
        _log.error("Failed to save state: %s", e)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _signal_handler(signum: int, frame: Any) -> None:
    """Graceful shutdown: let the current phase finish, then stop."""
    global _SHUTDOWN_REQUESTED
    sig_name = signal.Signals(signum).name
    _log.warning("SIGNAL %s received — will stop after current phase finishes.", sig_name)
    _log.warning("Press Ctrl+C again to force-kill.")
    _SHUTDOWN_REQUESTED = True
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def validate_environment() -> None:
    """Check all critical paths before starting. Raises RuntimeError on failure."""
    errors = []
    if not TRAIN_SCRIPT.exists():
        errors.append(f"Training script not found: {TRAIN_SCRIPT}")
    if not os.path.isfile(CLASS_MAPPING):
        errors.append(f"Class mapping not found: {CLASS_MAPPING}")
    if not os.path.isfile(LABELS_ALL):
        errors.append(f"Labels file not found: {LABELS_ALL}")
    if not os.path.isfile(LABELS_150):
        errors.append(f"Labels file (150) not found: {LABELS_150}")
    if errors:
        raise RuntimeError(
            "Environment validation failed:\n  " + "\n  ".join(errors)
        )
    _log.info("Environment validation: OK")


def check_disk_space(path: str) -> None:
    """Warn if disk space is critically low."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < MIN_DISK_GB:
            _log.warning(
                "Only %.1f GB free on %s. Training may fail if disk fills up!", free_gb, path
            )
        else:
            _log.info("Disk space: %.1f GB free", free_gb)
    except OSError:
        pass


def validate_checkpoint(ckpt_path: str) -> bool:
    """Check that checkpoint file exists and has a reasonable size."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return False
    size = os.path.getsize(ckpt_path)
    if size < MIN_CKPT_SIZE_BYTES:
        _log.warning(
            "Checkpoint %s is suspiciously small (%d bytes), skipping.", ckpt_path, size
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_state: ExperimentState,
    exp_root: str,
    state_file: str,
    all_states: Dict[str, ExperimentState],
    dry_run: bool = False,
    retry_failed: bool = False,
    phases: List[str] = None, # <--- ДОБАВИЛИ ЭТОТ ПАРАМЕТР
) -> bool:
    global _SHUTDOWN_REQUESTED
    
    if phases is None:
        phases = ["phase1", "phase2", "phase3"] # По умолчанию полный пайплайн
        
    exp_t0 = time.monotonic()

    for phase in phases: # <--- ТЕПЕРЬ ЦИКЛ ИДЕТ ПО ПЕРЕДАННОМУ СПИСКУ
        if _SHUTDOWN_REQUESTED:
            _log.info("[%s] Shutdown requested, stopping before %s.", exp_state.name, phase)
            save_state(state_file, all_states)
            return False

        status = exp_state.phase_status.get(phase, "pending")
        if status == "done":
            _log.info("[%s] %s: already done, skipping.", exp_state.name, phase)
            continue
        if status == "failed" and not retry_failed:
            _log.info(
                "[%s] %s: previously failed, skipping (use --retry_failed to retry).",
                exp_state.name, phase,
            )
            return False

        phase_dir = os.path.join(exp_root, exp_state.name, phase)
        exp_state.phase_dirs[phase] = phase_dir

        # ---- resolve previous phase checkpoint ----
        prev_ckpt: Optional[str] = None
        prev_phase_idx = phases.index(phase) - 1
        if prev_phase_idx >= 0:
            prev_phase = phases[prev_phase_idx]
            prev_ckpt = exp_state.phase_checkpoints.get(prev_phase) or ""
            if not validate_checkpoint(prev_ckpt):
                prev_dir = exp_state.phase_dirs.get(prev_phase, "")
                prev_ckpt = find_best_checkpoint(prev_dir) if prev_dir else None
            if prev_ckpt and not validate_checkpoint(prev_ckpt):
                prev_ckpt = None
            if not prev_ckpt:
                _log.warning(
                    "[%s] %s: no valid checkpoint from %s, training from scratch.",
                    exp_state.name, phase, prev_phase,
                )

        cmd = build_phase_cmd(exp_state.grid_params, phase, phase_dir, prev_ckpt)

        if dry_run:
            _log.info("\n[DRY RUN] %s / %s", exp_state.name, phase)
            _log.info("  Output: %s", phase_dir)
            if prev_ckpt:
                _log.info("  Load weights: %s", prev_ckpt)
            _log.info("  CMD: %s", " ".join(cmd))
            exp_state.phase_status[phase] = "dry_run"
            continue

        exp_state.phase_status[phase] = "running"
        save_state(state_file, all_states)

        try:
            rc = run_phase(cmd, phase_dir, f"{exp_state.name}/{phase}")
        except Exception as e:
            rc = -1
            _log.exception("[%s] %s: EXCEPTION during run_phase: %s", exp_state.name, phase, e)

        if rc != 0:
            exp_state.phase_status[phase] = "failed"
            save_state(state_file, all_states)
            _log.error("[%s] %s: FAILED (exit code %d)", exp_state.name, phase, rc)
            return False

        best_ckpt = find_best_checkpoint(phase_dir)
        if best_ckpt and not validate_checkpoint(best_ckpt):
            _log.warning(
                "[%s] %s: best checkpoint is invalid — trying last.ckpt fallback",
                exp_state.name, phase,
            )
            fallback = os.path.join(phase_dir, "checkpoints", "last.ckpt")
            best_ckpt = fallback if validate_checkpoint(fallback) else None

        if not best_ckpt:
            _log.warning(
                "[%s] %s: no valid checkpoint produced! Next phase will train from scratch.",
                exp_state.name, phase,
            )

        exp_state.phase_checkpoints[phase] = best_ckpt or ""
        exp_state.phase_status[phase] = "done"
        save_state(state_file, all_states)
        elapsed = time.monotonic() - exp_t0
        _log.info(
            "[%s] %s: DONE  ckpt=%s  (experiment elapsed: %.0fs)",
            exp_state.name, phase, best_ckpt, elapsed,
        )

    return True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_states: Dict[str, ExperimentState]) -> None:
    _log.info("=" * 80)
    _log.info("GRID SEARCH SUMMARY")
    _log.info("=" * 80)
    for name, st in sorted(all_states.items()):
        statuses = " | ".join(
            f"{p}: {st.phase_status.get(p, 'pending'):7s}" for p in ["phase1", "phase2", "phase3"]
        )
        _log.info("  %-50s  %s", name, statuses)
    total = len(all_states)
    phases = ["phase1", "phase2", "phase3"]
    done = sum(1 for s in all_states.values()
               if all(s.phase_status.get(p) == "done" for p in phases))
    failed = sum(1 for s in all_states.values()
                 if any(s.phase_status.get(p) == "failed" for p in phases))
    running = sum(1 for s in all_states.values()
                  if any(s.phase_status.get(p) == "running" for p in phases))
    pending = total - done - failed - running
    _log.info(
        "Total: %d  |  Done: %d  |  Failed: %d  |  Running: %d  |  Pending: %d",
        total, done, failed, running, pending,
    )
    if _SHUTDOWN_REQUESTED:
        _log.info("NOTE: Grid search was interrupted by user signal.")
    _log.info("=" * 80)


# ---------------------------------------------------------------------------
# Curated configs
# ---------------------------------------------------------------------------

def get_curated_configs() -> List[Dict[str, Any]]:
    """
    A/B Test: DINOv2 vs BEiTv2 on the 5 most promising ArcFace & Attention setups.
    Includes testing of the 'simple' neck (no normalization bottleneck).
    """
    
    # Железный фундамент
    base_params = {
        "embedding_dim": 512,
        "pooling_type": "attention",
        "use_cls_token": True,
        "head_type": "subcenter",
        "metric_loss_type": "circle",
    }

    # 5 архетипов гиперпараметров
    archetypes = [
        # 1. "The Champion" (Мягкий scale 30, жесткий margin 0.35, lnneck)
        {
            "num_attention_heads": 3, "neck_type": "lnneck", 
            "arcface_s": 30.0, "arcface_m": 0.35, "arcface_K": 3, "label_smoothing": 0.05
        },
        {
            "num_attention_heads": 3, "neck_type": "simple", 
            "arcface_s": 30.0, "arcface_m": 0.35, "arcface_K": 3, "label_smoothing": 0.05
        },
        {
            "num_attention_heads": 3, "neck_type": "simple", 
            "arcface_s": 30.0, "arcface_m": 0.35, "arcface_K": 1, "label_smoothing": 0.05
        },
        {
            "num_attention_heads": 3, "neck_type": "mlp", 
            "arcface_s": 30.0, "arcface_m": 0.35, "arcface_K": 3, "label_smoothing": 0.05
        },
        # 2. "Super-Focus" (1 голова бьет точно в цель, bnneck)
        {
            "num_attention_heads": 1, "neck_type": "bnneck", 
            "arcface_s": 30.0, "arcface_m": 0.2, "arcface_K": 3, "label_smoothing": 0.1
        },
        {
            "num_attention_heads": 1, "neck_type": "simple", 
            "arcface_s": 30.0, "arcface_m": 0.2, "arcface_K": 3, "label_smoothing": 0.1
        },
        {
            "num_attention_heads": 1, "neck_type": "mlp", 
            "arcface_s": 30.0, "arcface_m": 0.2, "arcface_K": 3, "label_smoothing": 0.1
        },
        # 3. "The Classic Baseline" (ArcFace K=1, 2 головы, bnneck)
        {
            "num_attention_heads": 2, "neck_type": "bnneck", 
            "arcface_s": 64.0, "arcface_m": 0.2, "arcface_K": 1, "label_smoothing": 0.1
        },
        # 4. "High Penalty + High K" (Сильное расталкивание m=0.35, lnneck)
        {
            "num_attention_heads": 2, "neck_type": "lnneck", 
            "arcface_s": 64.0, "arcface_m": 0.35, "arcface_K": 3, "label_smoothing": 0.05
        },
        # 5. "The Minimalist" (НОВОЕ: проверяем отсутствие нормализации)
        # Оставляем классические параметры ArcFace, чтобы кристально чисто оценить влияние 'simple' neck
        {
            "num_attention_heads": 2, "neck_type": "simple", 
            "arcface_s": 64.0, "arcface_m": 0.2, "arcface_K": 3, "label_smoothing": 0.1
        }
    ]

    backbones = [
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "beitv2_base_patch16_224.in1k_ft_in22k_in1k"
    ]

    configs = []
    
    # Скрещиваем каждый архетип с каждым бэкбоном
    for arch in archetypes:
        for backbone in backbones:
            cfg = copy.deepcopy(base_params)
            cfg.update(arch)
            cfg["backbone_model_name"] = backbone
            configs.append(cfg)

    return configs

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3-Phase Grid Search for Fish Classification")
    parser.add_argument(
        "--mode", choices=["full", "random", "resume", "curated"],
        default="curated",
        help="full=all combos, random=N random, resume=continue from state, curated=hand-picked best configs",
    )
    parser.add_argument("--n", type=int, default=8, help="Number of random configs (mode=random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mode=random)")
    parser.add_argument(
        "--output_root", type=str,
        default=os.path.join(BASE_OUTPUT_DIR, f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Root directory for all experiments",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments whose output_dir already exists")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Re-run phases that previously failed instead of skipping them")
    parser.add_argument(
        "--start_from", type=int, default=0,
        help="Start from experiment index (useful for resuming after crash)",
    )
    parser.add_argument("--fast_screen", action="store_true", 
                        help="Run ONLY Phase 1 for 10 epochs to quickly filter out bad configs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _SHUTDOWN_REQUESTED
    args = parse_args()

    exp_root = args.output_root
    os.makedirs(exp_root, exist_ok=True)

    # --- ЛОГИКА FAST SCREEN ---
    active_phases = ["phase1", "phase2", "phase3"]
    if args.fast_screen:
        _log.info("🚀 FAST SCREEN MODE ACTIVATED: Running only Phase 1 for 12 epochs!")
        active_phases = ["phase1"]
        PHASE_OVERRIDES["phase1"]["max_epochs"] = 12
        PHASE_OVERRIDES["phase1"]["use_swa"] = False
        PHASE_OVERRIDES["phase1"]["freeze_backbone_epochs"] = 1 # Чуть быстрее размораживаем
        PHASE_OVERRIDES["phase2"]["max_epochs"] = 12
        PHASE_OVERRIDES["phase2"]["use_swa"] = False
        PHASE_OVERRIDES["phase2"]["freeze_backbone_epochs"] = 1 # Чуть быстрее размораживаем
        PHASE_OVERRIDES["phase3"]["max_epochs"] = 12
        PHASE_OVERRIDES["phase3"]["use_swa"] = False
        PHASE_OVERRIDES["phase3"]["freeze_backbone_epochs"] = 1 # Чуть быстрее размораживаем
    # --------------------------


    setup_logging(exp_root)

    if not args.dry_run:
        validate_environment()
        install_signal_handlers()

    state_file = os.path.join(exp_root, "grid_search_state.json")
    check_disk_space(exp_root)

    # ---- build / load experiment list ----
    if args.mode == "resume":
        all_states = load_state(state_file)
        if not all_states:
            _log.error("No previous state found. Run with --mode full/random/curated first.")
            return
    else:
        if args.mode == "full":
            configs = generate_grid_combinations(SEARCH_SPACE)
        elif args.mode == "random":
            configs = sample_random_configs(SEARCH_SPACE, args.n, args.seed)
        elif args.mode == "curated":
            configs = get_curated_configs()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        all_states = load_state(state_file) if os.path.exists(state_file) else {}
        for i, cfg in enumerate(configs):
            name = make_experiment_name(cfg, i)
            if name not in all_states:
                all_states[name] = ExperimentState(
                    name=name,
                    grid_params=cfg,
                    created_at=datetime.now().isoformat(),
                )

    # Persist config snapshots for reproducibility
    with open(os.path.join(exp_root, "search_space.json"), "w") as f:
        json.dump(SEARCH_SPACE, f, indent=2)
    with open(os.path.join(exp_root, "fixed_params.json"), "w") as f:
        json.dump(FIXED_PARAMS, f, indent=2)
    with open(os.path.join(exp_root, "phase_overrides.json"), "w") as f:
        json.dump(PHASE_OVERRIDES, f, indent=2)

    save_state(state_file, all_states)

    _log.info("Grid Search Root: %s", exp_root)
    _log.info("Total experiments: %d", len(all_states))
    _log.info("Mode: %s", args.mode)
    if args.retry_failed:
        _log.info("Retry mode: previously failed phases will be re-run")
    if args.dry_run:
        _log.info("*** DRY RUN — no training will be executed ***")

    grid_t0 = time.monotonic()
    completed_count = 0
    failed_count = 0

    sorted_names = sorted(all_states.keys())
    for idx, name in enumerate(sorted_names):
        if _SHUTDOWN_REQUESTED:
            _log.info("Shutdown requested — stopping at experiment %d/%d.", idx, len(sorted_names))
            break

        if idx < args.start_from:
            continue

        exp = all_states[name]
        all_done = all(exp.phase_status.get(p) == "done" for p in ["phase1", "phase2", "phase3"])
        if all_done and args.skip_existing:
            _log.info("[%d/%d] %s: all phases done, skipping.", idx, len(sorted_names), name)
            completed_count += 1
            continue

        _log.info("#" * 70)
        _log.info("  EXPERIMENT [%d/%d]: %s", idx, len(sorted_names), name)
        _log.info("  Grid params: %s", json.dumps(exp.grid_params, indent=4))
        _log.info("#" * 70)

        try:
            success = run_experiment(
                exp, exp_root, state_file, all_states,
                dry_run=args.dry_run, retry_failed=args.retry_failed,
            )
        except Exception:
            success = False
            _log.exception("UNEXPECTED ERROR in experiment %s", name)
            for p in ["phase1", "phase2", "phase3"]:
                if exp.phase_status.get(p) == "running":
                    exp.phase_status[p] = "failed"
            save_state(state_file, all_states)

        if success:
            completed_count += 1
        elif not args.dry_run:
            failed_count += 1
            _log.info("Experiment %s had a failure. Continuing to next experiment...", name)

    grid_elapsed = time.monotonic() - grid_t0
    _log.info(
        "Grid search wall time: %.0fs (%.1fh)", grid_elapsed, grid_elapsed / 3600
    )
    _log.info("Completed this run: %d  |  Failed this run: %d", completed_count, failed_count)
    print_summary(all_states)


if __name__ == "__main__":
    main()

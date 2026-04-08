#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Autonomous Training Orchestrator for Fish Classification.

Launches sequential training runs unattended.  Between runs it analyses the
finished (or killed) metrics, auto-tunes hyperparameters, and starts a new
run from the best checkpoint.  Keeps going until:

* target validation accuracy is reached, OR
* no improvement for ``--patience_runs`` consecutive runs, OR
* ``--max_runs`` is exhausted.

During each run the orchestrator polls ``metrics_history.jsonl`` and will
**kill** the training process early when it detects:

* no val-accuracy improvement for ``--kill_patience_epochs`` epochs,
* train–val gap exceeds ``--max_overfit_gap``,
* val accuracy declining steadily.

Usage
-----
::

    python auto_train_loop.py \\
        --base_output_dir /home/fishial/Fishial/Experiments/v10 \\
        --max_runs 20 \\
        --target_val_accuracy 0.975 \\
        -- \\
        --dataset_name classification_v0.10_train \\
        --backbone_model_name beitv2_base_patch16_224.in1k_ft_in22k_in1k \\
        --output_dir /home/fishial/Fishial/Experiments/v10 \\
        ... (all lightning_train.py arguments for the first run)

Everything after ``--`` is forwarded to ``lightning_train.py`` for run #1.
Subsequent runs are derived automatically via ``AutoTuner``.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project path setup (same logic as lightning_train.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELIMITER = "fish-identification"
_pos = SCRIPT_DIR.find(DELIMITER)
if _pos != -1:
    _project_root = SCRIPT_DIR[: _pos + len(DELIMITER)]
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "lightning_train.py")

from module.classification_package.src.auto_tuner import (
    AutoTuner,
    TrainingDiagnosis,
    find_best_checkpoint,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("auto_train")


# ============================================================================
# Training monitor — watches metrics_history.jsonl of a live run
# ============================================================================

class TrainingMonitor:
    """Polls ``metrics_history.jsonl`` and decides whether to kill a run."""

    def __init__(
        self,
        metrics_path: str,
        kill_patience_epochs: int = 15,
        max_overfit_gap: float = 0.06,
        gap_growth_patience: int = 8,
        min_epochs_before_kill: int = 10,
    ):
        self.metrics_path = metrics_path
        self.kill_patience_epochs = kill_patience_epochs
        self.max_overfit_gap = max_overfit_gap
        self.gap_growth_patience = gap_growth_patience
        self.min_epochs_before_kill = min_epochs_before_kill

    # ------------------------------------------------------------------ io

    def read_all_metrics(self) -> List[Dict]:
        if not os.path.exists(self.metrics_path):
            return []
        metrics: List[Dict] = []
        try:
            with open(self.metrics_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
        except Exception:
            pass
        return metrics

    # ---------------------------------------------------------- decisions

    def should_kill(self) -> Tuple[bool, str]:
        metrics = self.read_all_metrics()
        if not metrics:
            return False, ""

        val_accs = [
            float(m["val/accuracy_epoch"])
            for m in metrics
            if "val/accuracy_epoch" in m
        ]
        train_accs = [
            float(m["train/accuracy_epoch"])
            for m in metrics
            if "train/accuracy_epoch" in m
        ]

        if not val_accs:
            return False, ""

        n = len(val_accs)
        if n < self.min_epochs_before_kill:
            return False, ""

        best_val = max(val_accs)
        best_epoch = val_accs.index(best_val)
        epochs_since_best = n - 1 - best_epoch

        # 1. Stagnation
        if epochs_since_best >= self.kill_patience_epochs:
            return True, (
                f"No val improvement for {epochs_since_best} epochs "
                f"(best={best_val:.4f} @ epoch {best_epoch})"
            )

        # 2. Overfitting gap too large
        if train_accs and len(train_accs) >= 3:
            w = min(3, len(val_accs))
            gap = sum(train_accs[-w:]) / w - sum(val_accs[-w:]) / w
            if gap > self.max_overfit_gap:
                return True, (
                    f"Overfit gap {gap:.4f} > limit {self.max_overfit_gap}"
                )

        # 3. Consistent gap growth
        if train_accs and len(val_accs) >= self.gap_growth_patience:
            gaps = [
                train_accs[i] - val_accs[i]
                for i in range(max(0, len(val_accs) - self.gap_growth_patience), len(val_accs))
                if i < len(train_accs)
            ]
            if len(gaps) >= self.gap_growth_patience and all(
                gaps[j] > gaps[j - 1] + 1e-4 for j in range(1, len(gaps))
            ):
                return True, (
                    f"Overfit gap growing for {self.gap_growth_patience} "
                    f"consecutive epochs ({gaps[0]:.4f}→{gaps[-1]:.4f})"
                )

        # 4. Steady val decline
        decline_window = 10
        if n >= decline_window:
            tail = val_accs[-decline_window:]
            if all(tail[j] < tail[j - 1] - 0.0005 for j in range(1, len(tail))):
                return True, (
                    f"Val accuracy declining for {decline_window} consecutive "
                    f"epochs ({tail[0]:.4f}→{tail[-1]:.4f})"
                )

        return False, ""

    # ---------------------------------------------------------- summary

    def get_summary(self) -> Dict:
        metrics = self.read_all_metrics()
        if not metrics:
            return {}
        val_accs = [
            float(m["val/accuracy_epoch"])
            for m in metrics
            if "val/accuracy_epoch" in m
        ]
        train_accs = [
            float(m["train/accuracy_epoch"])
            for m in metrics
            if "train/accuracy_epoch" in m
        ]
        if not val_accs:
            return {"epochs": len(metrics)}
        gap = (train_accs[-1] - val_accs[-1]) if train_accs else None
        return {
            "epochs": len(val_accs),
            "best_val_acc": max(val_accs),
            "best_val_epoch": val_accs.index(max(val_accs)),
            "final_val_acc": val_accs[-1],
            "final_train_acc": train_accs[-1] if train_accs else None,
            "overfit_gap": gap,
        }


# ============================================================================
# Autonomous trainer — the outer loop
# ============================================================================

class AutonomousTrainer:
    """Runs, monitors, kills, auto-tunes, repeats."""

    def __init__(
        self,
        base_output_dir: str,
        initial_train_args: List[str],
        max_runs: int = 20,
        target_val_accuracy: float = 0.975,
        patience_runs: int = 4,
        kill_patience_epochs: int = 15,
        max_overfit_gap: float = 0.06,
        min_epochs_before_kill: int = 10,
        poll_interval_seconds: int = 60,
        python_executable: Optional[str] = None,
        continue_from: Optional[str] = None,
    ):
        self.base_output_dir = base_output_dir
        self.initial_train_args = initial_train_args
        self.max_runs = max_runs
        self.target_val_accuracy = target_val_accuracy
        self.patience_runs = patience_runs
        self.kill_patience_epochs = kill_patience_epochs
        self.max_overfit_gap = max_overfit_gap
        self.min_epochs_before_kill = min_epochs_before_kill
        self.poll_interval = poll_interval_seconds
        self.python_exe = python_executable or sys.executable
        self.continue_from = continue_from

        self.run_history: List[Dict] = []
        self.best_val_accuracy_overall = 0.0
        self.best_checkpoint_path: Optional[str] = None
        self.runs_without_improvement = 0

        self.master_log_path = os.path.join(base_output_dir, "auto_train_log.jsonl")
        os.makedirs(base_output_dir, exist_ok=True)

        self._current_process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------ io

    def _log_run(self, info: Dict) -> None:
        with open(self.master_log_path, "a") as fh:
            fh.write(json.dumps(info, default=str) + "\n")

    # -------------------------------------------------------- directories

    def _find_latest_run_dir(self) -> Optional[str]:
        if not os.path.isdir(self.base_output_dir):
            return None
        subdirs = [
            os.path.join(self.base_output_dir, d)
            for d in os.listdir(self.base_output_dir)
            if os.path.isdir(os.path.join(self.base_output_dir, d))
            and os.path.exists(
                os.path.join(self.base_output_dir, d, "config.json")
            )
        ]
        if not subdirs:
            return None
        subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return subdirs[0]

    def _detect_new_run_dir(self, known_dirs: set, timeout: int = 300) -> Optional[str]:
        """Wait until a new run sub-directory appears."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.isdir(self.base_output_dir):
                for name in os.listdir(self.base_output_dir):
                    full = os.path.join(self.base_output_dir, name)
                    if full not in known_dirs and os.path.isdir(full):
                        cfg = os.path.join(full, "config.json")
                        if os.path.exists(cfg):
                            return full
            time.sleep(5)
        return None

    # ----------------------------------------------------- arg building

    def _config_to_cli_args(self, config: Dict) -> List[str]:
        """Convert a config dict to a flat list of CLI arguments."""
        args: List[str] = []
        skip = {
            "resume_from_checkpoint", "limit_train_batches", "limit_val_batches",
            "cache_dir", "no_cache", "auto_tune_from",
            "use_adaptive_regularization", "adaptive_gap_threshold",
            "adaptive_patience",
        }
        for key in sorted(config.keys()):
            if key in skip or key.startswith("_"):
                continue
            value = config[key]
            if value is None:
                continue
            if isinstance(value, bool):
                args.extend([f"--{key}", "true" if value else "false"])
            elif isinstance(value, list):
                args.extend([f"--{key}", ",".join(str(v) for v in value)])
            elif isinstance(value, float):
                args.extend([f"--{key}", f"{value:g}"])
            else:
                args.extend([f"--{key}", str(value)])
        # Always enable adaptive regularization in autonomous mode
        args.extend(["--use_adaptive_regularization", "true"])
        return args

    def _build_args_for_run(self, run_number: int) -> List[str]:
        """Build CLI args for the given run number."""

        if run_number == 0 and not self.continue_from:
            return list(self.initial_train_args)

        prev_run_dir: Optional[str] = None
        if run_number == 0 and self.continue_from:
            prev_run_dir = self.continue_from
        else:
            prev_run_dir = self._find_latest_run_dir()

        if prev_run_dir is None:
            logger.warning("No previous run found; falling back to initial args")
            return list(self.initial_train_args)

        tuner = AutoTuner(prev_run_dir)
        try:
            adjustments, diag, reasons = tuner.run(
                stale_runs=self.runs_without_improvement,
                best_val_overall=self.best_val_accuracy_overall,
            )
        except Exception as exc:
            logger.warning("Auto-tuning failed (%s); using previous config", exc)
            adjustments, reasons = {}, []

        prev_config = tuner.load_config()
        for k, v in adjustments.items():
            prev_config[k] = v

        # Always load from the best known checkpoint
        best_ckpt = self.best_checkpoint_path
        run_ckpt = find_best_checkpoint(prev_run_dir)
        if run_ckpt:
            if not (best_ckpt and os.path.exists(best_ckpt)):
                best_ckpt = run_ckpt
        if best_ckpt:
            prev_config["load_weights_from_checkpoint"] = best_ckpt
        prev_config.pop("resume_from_checkpoint", None)

        prev_config["output_dir"] = self.base_output_dir

        if reasons:
            logger.info(
                "Auto-tuning for run %d (stale_runs=%d):",
                run_number + 1, self.runs_without_improvement,
            )
            for r in reasons:
                logger.info("  → %s", r)
        else:
            logger.warning(
                "Auto-tuner proposed NO changes for run %d — this should not happen "
                "with stale_runs=%d", run_number + 1, self.runs_without_improvement,
            )

        return self._config_to_cli_args(prev_config)

    # ------------------------------------------------- process management

    def _kill_process(self, proc: subprocess.Popen) -> None:
        """Send SIGINT (allows Lightning to save checkpoint), then force-kill."""
        try:
            proc.send_signal(signal.SIGINT)
            logger.info("Sent SIGINT to training process (pid %d), waiting 120 s …", proc.pid)
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not exit after SIGINT; sending SIGKILL")
            proc.kill()
            proc.wait()

    # ---------------------------------------------------- single run

    def _run_training(self, run_number: int, args: List[str]) -> Tuple[int, Optional[str]]:
        """
        Launch ``lightning_train.py`` as a subprocess, monitor it, and
        optionally kill it early.  Returns ``(exit_code, run_dir)``.
        """

        cmd = [self.python_exe, TRAIN_SCRIPT] + args

        logger.info("=" * 70)
        logger.info(
            "RUN %d/%d  |  %s",
            run_number + 1,
            self.max_runs,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        logger.info("=" * 70)

        # Snapshot existing sub-dirs so we can detect the new one
        known_dirs: set = set()
        if os.path.isdir(self.base_output_dir):
            known_dirs = {
                os.path.join(self.base_output_dir, d)
                for d in os.listdir(self.base_output_dir)
                if os.path.isdir(os.path.join(self.base_output_dir, d))
            }

        log_path = os.path.join(
            self.base_output_dir,
            f"run_{run_number + 1:03d}_{datetime.now():%Y%m%d_%H%M%S}.log",
        )

        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        self._current_process = proc

        logger.info("Process started (pid %d), log → %s", proc.pid, log_path)

        # --- Wait for the new run directory to appear ---
        run_dir = self._detect_new_run_dir(known_dirs, timeout=600)
        if run_dir:
            logger.info("Detected run directory: %s", run_dir)
        else:
            logger.warning("Could not detect run directory within timeout")

        metrics_path = os.path.join(run_dir, "metrics_history.jsonl") if run_dir else None
        monitor = (
            TrainingMonitor(
                metrics_path,
                kill_patience_epochs=self.kill_patience_epochs,
                max_overfit_gap=self.max_overfit_gap,
                min_epochs_before_kill=self.min_epochs_before_kill,
            )
            if metrics_path
            else None
        )

        # --- Monitoring loop ---
        start_time = time.time()
        was_killed = False

        while proc.poll() is None:
            time.sleep(self.poll_interval)

            if monitor is None:
                continue

            summary = monitor.get_summary()
            elapsed = timedelta(seconds=int(time.time() - start_time))

            if summary.get("best_val_acc") is not None:
                gap_str = (
                    f"gap={summary['overfit_gap']:.4f}"
                    if summary.get("overfit_gap") is not None
                    else ""
                )
                logger.info(
                    "  Run %d | epoch %s | best_val=%.4f | cur_val=%.4f | %s | %s",
                    run_number + 1,
                    summary.get("epochs", "?"),
                    summary["best_val_acc"],
                    summary.get("final_val_acc", 0),
                    gap_str,
                    elapsed,
                )

            should_kill, reason = monitor.should_kill()
            if should_kill:
                logger.warning(
                    "KILLING run %d: %s", run_number + 1, reason
                )
                self._kill_process(proc)
                was_killed = True
                break

        exit_code = proc.returncode if proc.returncode is not None else -1
        self._current_process = None

        status = "KILLED" if was_killed else (
            "OK" if exit_code == 0 else f"FAILED (exit={exit_code})"
        )
        logger.info(
            "Run %d finished — %s  (elapsed %s)",
            run_number + 1,
            status,
            timedelta(seconds=int(time.time() - start_time)),
        )

        return exit_code, run_dir

    # ----------------------------------------------------- main loop

    def _seed_from_continue(self) -> None:
        """If --continue_from was provided, seed best_val / checkpoint."""
        if not self.continue_from:
            return
        try:
            tuner = AutoTuner(self.continue_from)
            diag = AutoTuner.diagnose(tuner.load_metrics())
            self.best_val_accuracy_overall = diag.best_val_accuracy
            ckpt = find_best_checkpoint(self.continue_from)
            if ckpt:
                self.best_checkpoint_path = ckpt
            logger.info(
                "Continuing from %s (best_val=%.4f, ckpt=%s)",
                self.continue_from,
                self.best_val_accuracy_overall,
                self.best_checkpoint_path,
            )
        except Exception as exc:
            logger.warning("Could not seed from --continue_from: %s", exc)

    def run(self) -> None:
        """Top-level orchestration loop."""

        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  AUTONOMOUS TRAINING ORCHESTRATOR                                  ║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info("║  Base dir          : %s", self.base_output_dir)
        logger.info("║  Max runs          : %d", self.max_runs)
        logger.info("║  Target val acc    : %.4f", self.target_val_accuracy)
        logger.info("║  Patience (runs)   : %d", self.patience_runs)
        logger.info("║  Kill patience     : %d epochs", self.kill_patience_epochs)
        logger.info("║  Max overfit gap   : %.3f", self.max_overfit_gap)
        logger.info("║  Poll interval     : %d s", self.poll_interval)
        logger.info("╚" + "═" * 68 + "╝")

        self._seed_from_continue()

        overall_start = time.time()

        for run_idx in range(self.max_runs):
            # --- Build arguments ---
            try:
                args = self._build_args_for_run(run_idx)
            except Exception as exc:
                logger.error("Failed to build args for run %d: %s", run_idx + 1, exc)
                break

            # --- Launch & monitor ---
            exit_code, run_dir = self._run_training(run_idx, args)
            run_elapsed = timedelta(
                seconds=int(time.time() - overall_start)
            )

            # --- Analyse results ---
            run_info: Dict = {
                "run_number": run_idx + 1,
                "run_dir": run_dir,
                "exit_code": exit_code,
                "timestamp": datetime.now().isoformat(),
            }

            if (
                run_dir
                and os.path.exists(os.path.join(run_dir, "metrics_history.jsonl"))
            ):
                try:
                    tuner = AutoTuner(run_dir)
                    diag = AutoTuner.diagnose(tuner.load_metrics())

                    run_info.update(
                        {
                            "best_val_accuracy": diag.best_val_accuracy,
                            "best_val_epoch": diag.best_val_epoch,
                            "final_val_accuracy": diag.final_val_accuracy,
                            "final_train_accuracy": diag.final_train_accuracy,
                            "overfit_gap": round(diag.overfit_gap, 5),
                            "overfitting_severity": diag.overfitting_severity,
                            "total_epochs": diag.total_epochs,
                        }
                    )

                    # Did this run improve the global best?
                    if diag.best_val_accuracy > self.best_val_accuracy_overall:
                        delta = diag.best_val_accuracy - self.best_val_accuracy_overall
                        self.best_val_accuracy_overall = diag.best_val_accuracy
                        ckpt = find_best_checkpoint(run_dir)
                        if ckpt:
                            self.best_checkpoint_path = ckpt
                        self.runs_without_improvement = 0
                        logger.info(
                            "★ NEW BEST: %.4f (+%.4f)  checkpoint: %s",
                            self.best_val_accuracy_overall,
                            delta,
                            self.best_checkpoint_path,
                        )
                    else:
                        self.runs_without_improvement += 1
                        logger.info(
                            "No improvement (best=%.4f, stale runs %d/%d)",
                            self.best_val_accuracy_overall,
                            self.runs_without_improvement,
                            self.patience_runs,
                        )

                except Exception as exc:
                    logger.warning("Analysis of run %d failed: %s", run_idx + 1, exc)
            else:
                logger.warning(
                    "Run %d produced no metrics (exit_code=%d)", run_idx + 1, exit_code
                )
                run_info["error"] = "no_metrics"
                # Count a failed run toward patience so we don't loop forever
                self.runs_without_improvement += 1

            self.run_history.append(run_info)
            self._log_run(run_info)

            # --- Convergence checks ---
            if self.best_val_accuracy_overall >= self.target_val_accuracy:
                logger.info(
                    "🎯 TARGET REACHED: %.4f >= %.4f after %d run(s)!",
                    self.best_val_accuracy_overall,
                    self.target_val_accuracy,
                    run_idx + 1,
                )
                break

            if self.runs_without_improvement >= self.patience_runs:
                logger.info(
                    "CONVERGED: no improvement for %d consecutive runs.  "
                    "Best val accuracy: %.4f",
                    self.patience_runs,
                    self.best_val_accuracy_overall,
                )
                break

            logger.info("Pausing 30 s before next run …")
            time.sleep(30)

        # --- Final summary ---
        total = timedelta(seconds=int(time.time() - overall_start))
        self._print_summary(total)

    # ------------------------------------------------------------ summary

    def _print_summary(self, total_elapsed: timedelta) -> None:
        logger.info("")
        logger.info("╔" + "═" * 68 + "╗")
        logger.info("║  ORCHESTRATOR SUMMARY                                              ║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info("║  Total runs   : %d", len(self.run_history))
        logger.info("║  Total time   : %s", total_elapsed)
        logger.info("║  Best val acc : %.4f", self.best_val_accuracy_overall)
        logger.info("║  Best ckpt    : %s", self.best_checkpoint_path)
        logger.info("║  Target met   : %s", self.best_val_accuracy_overall >= self.target_val_accuracy)
        logger.info("╚" + "═" * 68 + "╝")

        # Per-run table
        logger.info("")
        logger.info(
            "  %-4s  %-10s  %-10s  %-8s  %-8s  %s",
            "Run", "BestVal", "FinalVal", "Gap", "Epochs", "Status",
        )
        logger.info("  " + "-" * 62)
        for ri in self.run_history:
            bv = ri.get("best_val_accuracy")
            fv = ri.get("final_val_accuracy")
            gap = ri.get("overfit_gap")
            ep = ri.get("total_epochs", "?")
            ec = ri.get("exit_code", "?")
            status = "OK" if ec == 0 else ("KILLED" if ec == -2 else f"exit={ec}")
            if ri.get("error"):
                status = ri["error"]
            logger.info(
                "  %-4d  %-10s  %-10s  %-8s  %-8s  %s",
                ri["run_number"],
                f"{bv:.4f}" if bv is not None else "—",
                f"{fv:.4f}" if fv is not None else "—",
                f"{gap:.4f}" if gap is not None else "—",
                ep,
                status,
            )
        logger.info("")

        # Persist
        summary_path = os.path.join(self.base_output_dir, "auto_train_summary.json")
        payload = {
            "total_runs": len(self.run_history),
            "total_time": str(total_elapsed),
            "best_val_accuracy": self.best_val_accuracy_overall,
            "best_checkpoint": self.best_checkpoint_path,
            "target_reached": self.best_val_accuracy_overall >= self.target_val_accuracy,
            "runs": self.run_history,
        }
        with open(summary_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Summary saved → %s", summary_path)
        logger.info("Master log   → %s", self.master_log_path)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous Training Orchestrator — run and forget.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (start from scratch):

  python auto_train_loop.py \\
      --base_output_dir /home/fishial/Fishial/Experiments/v10 \\
      --max_runs 20 \\
      --target_val_accuracy 0.975 \\
      -- \\
      --dataset_name classification_v0.10_train \\
      --output_dir /home/fishial/Fishial/Experiments/v10 \\
      --backbone_model_name beitv2_base_patch16_224.in1k_ft_in22k_in1k \\
      ... (all lightning_train.py args)

Example (continue from a previous run):

  python auto_train_loop.py \\
      --base_output_dir /home/fishial/Fishial/Experiments/v10 \\
      --continue_from /home/fishial/Fishial/Experiments/v10/beitv2_..._20260221_071359 \\
      --max_runs 20
        """,
    )

    parser.add_argument(
        "--base_output_dir", type=str, required=True,
        help="Base directory for all training runs.",
    )
    parser.add_argument(
        "--max_runs", type=int, default=20,
        help="Maximum number of sequential training runs (default: 20).",
    )
    parser.add_argument(
        "--target_val_accuracy", type=float, default=0.975,
        help="Stop when this validation accuracy is achieved (default: 0.975).",
    )
    parser.add_argument(
        "--patience_runs", type=int, default=4,
        help="Stop if no improvement for N consecutive runs (default: 4).",
    )
    parser.add_argument(
        "--kill_patience_epochs", type=int, default=15,
        help="Kill a run if val accuracy hasn't improved for N epochs (default: 15).",
    )
    parser.add_argument(
        "--max_overfit_gap", type=float, default=0.06,
        help="Kill a run if train−val gap exceeds this (default: 0.06).",
    )
    parser.add_argument(
        "--min_epochs_before_kill", type=int, default=10,
        help="Don't kill a run before this many epochs (default: 10).",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=60,
        help="Seconds between metric checks during a run (default: 60).",
    )
    parser.add_argument(
        "--python_exe", type=str, default=None,
        help="Python executable for lightning_train.py.  "
             "Defaults to the same Python running this script.",
    )
    parser.add_argument(
        "--continue_from", type=str, default=None,
        help="Path to a previous run directory (or parent dir).  "
             "The orchestrator will auto-tune from it for run #1 instead "
             "of using the -- arguments.  Great for resuming an "
             "interrupted orchestration.",
    )

    args, remaining = parser.parse_known_args()

    # Strip leading '--' separator
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]

    return args, remaining


# ============================================================================
# Entry
# ============================================================================

def main():
    args, train_args = parse_args()

    if not train_args and not args.continue_from:
        logger.error(
            "No training arguments provided.  Pass lightning_train.py args "
            "after '--', or use --continue_from to resume from a previous run."
        )
        sys.exit(1)

    orchestrator = AutonomousTrainer(
        base_output_dir=args.base_output_dir,
        initial_train_args=train_args,
        max_runs=args.max_runs,
        target_val_accuracy=args.target_val_accuracy,
        patience_runs=args.patience_runs,
        kill_patience_epochs=args.kill_patience_epochs,
        max_overfit_gap=args.max_overfit_gap,
        min_epochs_before_kill=args.min_epochs_before_kill,
        poll_interval_seconds=args.poll_interval,
        python_executable=args.python_exe,
        continue_from=args.continue_from,
    )

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Interrupted by user (Ctrl+C).")
        if orchestrator._current_process and orchestrator._current_process.poll() is None:
            logger.info("Stopping current training …")
            orchestrator._kill_process(orchestrator._current_process)
        logger.info(
            "Best val accuracy so far: %.4f", orchestrator.best_val_accuracy_overall
        )
        if orchestrator.best_checkpoint_path:
            logger.info("Best checkpoint: %s", orchestrator.best_checkpoint_path)
        orchestrator._print_summary(
            timedelta(seconds=0)
        )


if __name__ == "__main__":
    main()

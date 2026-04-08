# -*- coding: utf-8 -*-
"""
Host RAM snapshots for debugging OOM ("Killed") during training.

- Linux: RSS via /proc/<pid>/statm for the trainer process and its *direct* children
  (typical DataLoader workers are children of the main Python process).
- Optional psutil speeds up listing child PIDs when installed.
- CUDA allocated/reserved bytes are included when torch.cuda is available.

Logs: one JSON object per line in ``host_memory_log.jsonl`` under the run output_dir.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl

logger = logging.getLogger(__name__)


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 4096


def _rss_bytes_proc(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/statm", encoding="utf-8") as f:
            fields = f.read().split()
        return int(fields[1]) * _page_size()
    except (OSError, ValueError, IndexError, TypeError):
        return 0


def _child_pids_psutil(ppid: int) -> List[int]:
    try:
        import psutil

        p = psutil.Process(ppid)
        return [c.pid for c in p.children(recursive=False)]
    except Exception:
        return []


def _child_pids_scan_proc(ppid: int) -> List[int]:
    """Slow fallback: scan /proc (Linux)."""
    kids: List[int] = []
    try:
        for ent in os.listdir("/proc"):
            if not ent.isdigit():
                continue
            cpid = int(ent)
            try:
                with open(f"/proc/{cpid}/status", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            if int(line.split()[1]) == ppid:
                                kids.append(cpid)
                            break
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        pass
    return kids


def snapshot_host_memory(root_pid: Optional[int] = None) -> Dict[str, Any]:
    root_pid = root_pid or os.getpid()
    child_pids = _child_pids_psutil(root_pid)
    if not child_pids:
        child_pids = _child_pids_scan_proc(root_pid)

    main_rss = _rss_bytes_proc(root_pid)
    per_child: List[Dict[str, int]] = []
    children_rss = 0
    for cpid in child_pids:
        rss = _rss_bytes_proc(cpid)
        children_rss += rss
        per_child.append({"pid": cpid, "rss_bytes": rss})

    max_child_rss = max((c["rss_bytes"] for c in per_child), default=0)
    out: Dict[str, Any] = {
        "pid": root_pid,
        "main_rss_bytes": main_rss,
        "children_rss_bytes": children_rss,
        "children_max_rss_bytes": max_child_rss,
        "total_rss_bytes": main_rss + children_rss,
        "num_children": len(child_pids),
        "children": sorted(per_child, key=lambda x: -x["rss_bytes"])[:32],
        "rss_sum_note": "Summing child RSS over-counts shared RAM (fork/COW, mmap). Use children_max_rss_bytes or `smem -P python` for PSS.",
    }

    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            out["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception:
        pass

    return out


class HostMemoryMonitorCallback(pl.Callback):
    """Log host RSS after train / val epochs; append JSONL for post-mortem analysis."""

    def __init__(self, output_dir: str, enabled: bool = True):
        super().__init__()
        self.output_dir = output_dir
        self.enabled = enabled
        self._path = os.path.join(output_dir, "host_memory_log.jsonl")

    def _emit(self, trainer: pl.Trainer, pl_module: pl.LightningModule, phase: str) -> None:
        if not self.enabled:
            return
        snap = snapshot_host_memory()
        snap["time_utc"] = datetime.now(timezone.utc).isoformat()
        snap["epoch"] = int(getattr(trainer, "current_epoch", -1))
        snap["global_step"] = int(getattr(trainer, "global_step", -1))
        snap["phase"] = phase

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("host_memory_monitor: could not write %s: %s", self._path, e)

        g_main = snap["main_rss_bytes"] / 1e9
        g_ch = snap["children_rss_bytes"] / 1e9
        g_ch_max = snap.get("children_max_rss_bytes", 0) / 1e9
        g_tot = snap["total_rss_bytes"] / 1e9
        cuda_a = snap.get("cuda_allocated_bytes")
        cuda_msg = ""
        if cuda_a is not None:
            cuda_msg = f" cuda_alloc={cuda_a / 1e9:.2f} GB"
        logger.info(
            "Host RAM [%s ep=%s]: main=%.2f GB workers_sum=%.2f GB (inflated if fork) workers_max=%.2f GB "
            "naive_total=%.2f GB children=%d%s",
            phase,
            snap["epoch"],
            g_main,
            g_ch,
            g_ch_max,
            g_tot,
            snap["num_children"],
            cuda_msg,
        )
        top = snap["children"][:5]
        if top:
            logger.info(
                "  top worker RSS (sample): %s",
                ", ".join(f"pid={c['pid']} {c['rss_bytes'] / 1e9:.2f}GB" for c in top),
            )

        lg = trainer.logger
        if lg is not None:
            try:
                step = int(getattr(trainer, "global_step", snap["epoch"]))
                lg.log_metrics(
                    {
                        "host_mem/main_gb": g_main,
                        "host_mem/children_sum_gb": g_ch,
                        "host_mem/children_max_gb": g_ch_max,
                        "host_mem/naive_total_gb": g_tot,
                    },
                    step=step,
                )
            except Exception:
                pass

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._emit(trainer, pl_module, "train_start")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._emit(trainer, pl_module, "train_epoch_end")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._emit(trainer, pl_module, "val_epoch_end")

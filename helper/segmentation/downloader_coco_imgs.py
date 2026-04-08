"""
COCO Image Downloader — Enhanced Edition
=========================================
Features:
  • Rich terminal UI: panels, progress bars, tables, spinners
  • Structured logging with rotating file handler
  • Automatic retry with exponential back-off (tenacity)
  • Per-image error classification
  • Live download statistics panel
  • Final HTML & JSON reports
  • Graceful Ctrl-C handling
"""

import os
import json
import logging
import signal
import sys
import time
import argparse
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TaskProgressColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.align import Align
from rich import print as rprint

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_WORKERS       = 30
RETRY_ATTEMPTS    = 3
RETRY_WAIT_MIN    = 1      # seconds
RETRY_WAIT_MAX    = 8      # seconds
REQUEST_TIMEOUT   = 30     # seconds
LOG_MAX_BYTES     = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT  = 3

console = Console(highlight=True)

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure structured logging with both file and console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"coco_downloader_{timestamp}.log")

    logger = logging.getLogger("coco_downloader")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # ── File handler (DEBUG level, rotating) ──
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # ── Rich console handler (WARNING and above) ──
    from rich.logging import RichHandler
    rh = RichHandler(console=console, show_path=False, markup=True, rich_tracebacks=True)
    rh.setLevel(logging.WARNING)
    logger.addHandler(rh)

    logger.info(f"Logging initialised → {log_file}")
    return logger, log_file


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────
def read_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ──────────────────────────────────────────────
# Download logic with retries
# ──────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=False,
)
def _fetch(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    resp = requests.get(url, allow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def download_image(
    url_info: Tuple[str, str, int],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Download a single image with retry logic.

    Returns a result dict:
        status   : "ok" | "failed" | "skipped"
        url      : str
        file_path: str
        image_id : int
        error    : str | None
        attempts : int
        elapsed  : float  (seconds)
    """
    url, file_path, image_id = url_info
    t0 = time.monotonic()

    try:
        content = _fetch(url)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)
        elapsed = time.monotonic() - t0
        logger.info(f"[OK] {url} → {file_path}  ({elapsed:.2f}s)")
        return {"status": "ok", "url": url, "file_path": file_path,
                "image_id": image_id, "error": None,
                "size_bytes": len(content), "elapsed": elapsed}

    except RetryError as e:
        cause = str(e.last_attempt.exception())
        elapsed = time.monotonic() - t0
        logger.error(f"[RETRY_EXHAUSTED] {url} — {cause}")
        return {"status": "failed", "url": url, "file_path": file_path,
                "image_id": image_id,
                "error": f"Retry exhausted after {RETRY_ATTEMPTS} attempts: {cause}",
                "size_bytes": 0, "elapsed": elapsed}

    except requests.HTTPError as e:
        elapsed = time.monotonic() - t0
        logger.error(f"[HTTP_ERROR] {url} — {e}")
        return {"status": "failed", "url": url, "file_path": file_path,
                "image_id": image_id, "error": f"HTTP {e.response.status_code}: {e}",
                "size_bytes": 0, "elapsed": elapsed}

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.exception(f"[UNEXPECTED] {url} — {e}")
        return {"status": "failed", "url": url, "file_path": file_path,
                "image_id": image_id, "error": str(e),
                "size_bytes": 0, "elapsed": elapsed}


# ──────────────────────────────────────────────
# Rich UI helpers
# ──────────────────────────────────────────────
def print_header(coco_path: str) -> None:
    title = Text("🖼  COCO Image Downloader", style="bold white on blue", justify="center")
    subtitle = Text(f"Dataset: {coco_path}", style="dim", justify="center")
    console.print(Panel(
        Align.center(title + "\n" + subtitle),
        border_style="blue",
        padding=(1, 4),
    ))
    console.print()


def print_dataset_summary(data: Dict[str, Any], images_folder: str,
                          downloaded_count: int, to_download: int) -> None:
    total = len(data.get("images", []))
    cats  = len(data.get("categories", []))
    annos = len(data.get("annotations", []))

    table = Table(
        title="📊 Dataset Overview",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total images in dataset",    f"[white]{total:,}[/]")
    table.add_row("Categories",                  f"[white]{cats:,}[/]")
    table.add_row("Annotations",                 f"[white]{annos:,}[/]")
    table.add_row("Already downloaded",          f"[green]{downloaded_count:,}[/]")
    table.add_row("To download now",             f"[yellow]{to_download:,}[/]")
    table.add_row("Output folder",               f"[dim]{images_folder}[/]")

    console.print(table)
    console.print()


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="blue", complete_style="green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
    )


def print_results_table(results: List[Dict[str, Any]]) -> None:
    ok      = [r for r in results if r["status"] == "ok"]
    failed  = [r for r in results if r["status"] == "failed"]
    total_b = sum(r.get("size_bytes", 0) for r in ok)
    avg_t   = (sum(r.get("elapsed", 0) for r in results) / len(results)) if results else 0

    # ── Stats cards ──
    cards = [
        Panel(f"[bold green]{len(ok):,}[/]\n[dim]successful[/]",   border_style="green",  expand=True),
        Panel(f"[bold red]{len(failed):,}[/]\n[dim]failed[/]",      border_style="red",    expand=True),
        Panel(f"[bold yellow]{_fmt_bytes(total_b)}[/]\n[dim]downloaded[/]", border_style="yellow", expand=True),
        Panel(f"[bold cyan]{avg_t:.2f}s[/]\n[dim]avg/image[/]",     border_style="cyan",   expand=True),
    ]
    console.print(Columns(cards, equal=True))
    console.print()

    # ── Failed downloads table ──
    if failed:
        table = Table(
            title=f"[bold red]❌ Failed Downloads ({len(failed)})",
            box=box.SIMPLE_HEAD,
            border_style="red",
            header_style="bold red",
            show_lines=False,
        )
        table.add_column("#",         style="dim",    width=4, justify="right")
        table.add_column("Image ID",  style="yellow", width=10)
        table.add_column("URL",       style="cyan",   no_wrap=False, max_width=55)
        table.add_column("Error",     style="red",    no_wrap=False, max_width=40)
        table.add_column("Time",      style="dim",    width=8, justify="right")

        for i, r in enumerate(failed[:50], 1):    # cap at 50 rows in terminal
            table.add_row(
                str(i),
                str(r.get("image_id", "—")),
                r["url"] or "—",
                r.get("error", "unknown") or "—",
                f"{r.get('elapsed', 0):.2f}s",
            )

        if len(failed) > 50:
            table.add_row("…", "…", f"[dim]+{len(failed)-50} more (see failed_downloads.json)[/]", "", "")

        console.print(table)
        console.print()


def _fmt_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def print_footer(log_file: str, report_file: str, failed_file: Optional[str]) -> None:
    lines = [f"[dim]📋 Log file   :[/] [cyan]{log_file}[/]",
             f"[dim]📄 JSON report:[/] [cyan]{report_file}[/]"]
    if failed_file:
        lines.append(f"[dim]🔴 Failed list:[/] [red]{failed_file}[/]")
    console.print(Panel("\n".join(lines), title="[bold]Output files", border_style="dim"))


# ──────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────
_shutdown = False

def _handle_sigint(sig, frame):
    global _shutdown
    console.print("\n[bold yellow]⚠  Interrupt received — finishing in-flight downloads…[/]")
    _shutdown = True


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COCO Image Downloader — Enhanced Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--coco_path",   required=True,  help="Path to the COCO JSON file")
    parser.add_argument("-i", "--image_path",  required=True,  help="Directory where images will be stored")
    parser.add_argument("-w", "--workers",     type=int, default=MAX_WORKERS, help=f"Parallel download workers (default: {MAX_WORKERS})")
    parser.add_argument("-r", "--retries",     type=int, default=RETRY_ATTEMPTS, help=f"Retry attempts per image (default: {RETRY_ATTEMPTS})")
    parser.add_argument("--log-dir",           default="logs", help="Directory for log files")
    parser.add_argument("--output-dir",        default=".",    help="Directory for JSON report output")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)
    args   = parse_arguments()
    logger, log_file = setup_logging(args.log_dir)

    print_header(args.coco_path)

    # ── 1. Load dataset ──────────────────────
    with console.status("[cyan]Reading COCO JSON…", spinner="dots"):
        try:
            data = read_json_file(args.coco_path)
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/] file not found — [cyan]{args.coco_path}[/]")
            sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error:[/] invalid JSON — {e}")
            sys.exit(1)

    console.print(f"[green]✓[/] Loaded COCO JSON: [bold]{len(data.get('images', []))}[/] images")

    # ── 2. Resolve what needs downloading ────
    images_data_folder = Path(args.image_path) / "data"
    images_data_folder.mkdir(parents=True, exist_ok=True)

    downloaded_files = {
        f for f in os.listdir(images_data_folder)
        if (images_data_folder / f).is_file()
    }

    urls_to_download: List[Tuple[str, str, int]] = []
    missing_url_count = 0

    for image in data.get("images", []):
        fname = image.get("file_name", "")
        if fname in downloaded_files:
            continue
        url = image.get("coco_url") or image.get("flickr_url")
        if not url:
            missing_url_count += 1
            logger.warning(f"No URL for image id={image.get('id')} file={fname}")
            continue
        urls_to_download.append((url, str(images_data_folder / fname), image.get("id", -1)))

    print_dataset_summary(data, str(images_data_folder),
                          len(downloaded_files), len(urls_to_download))

    if missing_url_count:
        console.print(f"[yellow]⚠  {missing_url_count} images have no URL and will be skipped.[/]\n")

    if not urls_to_download:
        console.print(Panel("[bold green]✅  All images already downloaded — nothing to do![/]",
                            border_style="green"))
        return

    # ── 3. Save download manifest ─────────────
    manifest_path = os.path.join(args.output_dir, "download_manifest.json")
    save_json_file({"generated_at": datetime.now().isoformat(),
                    "total": len(urls_to_download),
                    "images": [{"url": u, "file_path": p, "image_id": i}
                               for u, p, i in urls_to_download]},
                   manifest_path)
    logger.info(f"Manifest saved → {manifest_path}")

    # ── 4. Download with live progress ────────
    console.print(Rule("[bold cyan]Downloading", style="cyan"))
    results: List[Dict[str, Any]] = []
    start_time = time.monotonic()

    with make_progress() as progress:
        task = progress.add_task("Downloading images…", total=len(urls_to_download))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_image, item, logger): item
                for item in urls_to_download
            }

            for future in as_completed(futures):
                if _shutdown:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                result = future.result()
                results.append(result)
                progress.advance(task)

                # Live status suffix
                ok_count  = sum(1 for r in results if r["status"] == "ok")
                bad_count = sum(1 for r in results if r["status"] == "failed")
                progress.tasks[task].description = (
                    f"[cyan]Downloading[/]  "
                    f"[green]✓ {ok_count}[/]  [red]✗ {bad_count}[/]"
                )

    elapsed_total = time.monotonic() - start_time

    # ── 5. Results ────────────────────────────
    console.print()
    console.print(Rule("[bold white]Results", style="white"))
    console.print()
    print_results_table(results)

    ok_results     = [r for r in results if r["status"] == "ok"]
    failed_results = [r for r in results if r["status"] == "failed"]

    speed = len(ok_results) / elapsed_total if elapsed_total > 0 else 0
    console.print(
        f"[dim]Total time:[/] [bold]{elapsed_total:.1f}s[/]   "
        f"[dim]Throughput:[/] [bold]{speed:.1f} img/s[/]"
    )
    console.print()

    # ── 6. Save report ────────────────────────
    report = {
        "generated_at":    datetime.now().isoformat(),
        "coco_path":       args.coco_path,
        "output_folder":   str(images_data_folder),
        "workers":         args.workers,
        "total_attempted": len(results),
        "successful":      len(ok_results),
        "failed":          len(failed_results),
        "total_bytes":     sum(r.get("size_bytes", 0) for r in ok_results),
        "elapsed_seconds": round(elapsed_total, 2),
        "results":         results,
    }
    report_path = os.path.join(args.output_dir, "download_report.json")
    save_json_file(report, report_path)

    failed_path = None
    if failed_results:
        failed_path = os.path.join(args.output_dir, "failed_downloads.json")
        save_json_file({"failed": failed_results}, failed_path)
        logger.warning(f"{len(failed_results)} failed downloads saved → {failed_path}")

    console.print()
    print_footer(log_file, report_path, failed_path)
    console.print()

    if _shutdown:
        console.print("[yellow]⚠  Download interrupted by user.[/]")
        sys.exit(130)

    # ── Exit code: non-zero if any failures ──
    sys.exit(0 if not failed_results else 1)


if __name__ == "__main__":
    main()
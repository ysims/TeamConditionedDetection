#!/usr/bin/env python3
"""
Runs every training config in configs/ sequentially (one GPU => one job
at a time makes more sense than contending for it), then the zero-shot
baselines and the params report, and prints a consolidated summary table
at the end pulled from each run's metrics/final.json. Continues past a
failing config rather than aborting the whole sweep - failures are
reported in the summary, not raised.

Usage:
    python3 scripts/run_all.py                       # everything in configs/
    python3 scripts/run_all.py --configs ball_baseline ball_film_late
    python3 scripts/run_all.py --skip-zero-shot --skip-params-report
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semdetect.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

# Cheapest/fastest configs first, so early results land sooner in a long
# sequential sweep; anything in configs/ not listed here (e.g. a config
# added later) is appended after, alphabetically.
DEFAULT_CONFIG_ORDER = [
    "ball_baseline",
    "ball_film_late",
    "ball_film_early",
    "yolov8_film",
    "ball_film_bert",
    "ball_film_e5",
    "ball_film_random_embedding",
    "ball_film_wrong_descriptor",
    "fcos_baseline",
    "fcos_film",
    "fasterrcnn_baseline",
    "fasterrcnn_film",
    "rtdetr_baseline",
    "rtdetr_film",
]

ZERO_SHOT_MODELS = ["grounding_dino", "owlvit"]


def run(cmd: list[str], log_path: Path, timeout_sec: float | None = None) -> int:
    """Runs cmd in its own process group so a timeout can kill the whole
    tree (DataLoader workers included), not just the direct child - a
    plain subprocess.run(timeout=...) only kills the immediate process and
    can leave orphaned workers behind.
    """
    print(f"$ {' '.join(cmd)}")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=REPO_ROOT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            hours = (timeout_sec or 0) / 3600
            print(f"  TIMEOUT after {hours:.1f}h with no exit - killing process group")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return -9


def summarize_training_run(config_path: Path) -> str:
    try:
        config = load_config(config_path)
        final_path = REPO_ROOT / config.train.output_dir / "metrics" / "final.json"
        if not final_path.exists():
            return "no metrics/final.json (run failed before first eval?)"
        import json

        final = json.loads(final_path.read_text())
        metric_name = config.train.checkpoint_metric
        best = final.get(f"best_{metric_name}")
        test = final.get("test", {}).get(metric_name)
        epochs = f"{final.get('epochs_run', '?')}/{final.get('epochs_configured', '?')}"
        early = " (stopped early)" if final.get("stopped_early") else ""
        return f"best val {metric_name}={best:.4f}  test {metric_name}={test:.4f}  epochs={epochs}{early}"
    except Exception as e:
        return f"couldn't summarize: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=None, help="Config names (no .yaml); default: everything in configs/")
    parser.add_argument("--test-manifest", default="data/manifest/test.csv")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-params-report", action="store_true")
    parser.add_argument("--log-dir", default="outputs/sweep_logs")
    parser.add_argument(
        "--job-timeout-hours",
        type=float,
        default=4.0,
        help="Kill and mark failed any single job (training or zero-shot) that runs longer than this. "
        "Guards against a hang silently blocking the rest of the sweep - e.g. a pathological NMS "
        "candidate count once stalled a training run for over a day. 0 disables the timeout.",
    )
    args = parser.parse_args()
    job_timeout = args.job_timeout_hours * 3600 if args.job_timeout_hours > 0 else None

    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.configs is not None:
        names = args.configs
    else:
        available = {p.stem for p in CONFIGS_DIR.glob("*.yaml")}
        names = [n for n in DEFAULT_CONFIG_ORDER if n in available]
        names += sorted(available - set(names))

    results = []
    t_start = time.time()

    for name in names:
        config_path = CONFIGS_DIR / f"{name}.yaml"
        if not config_path.exists():
            print(f"skipping {name}: {config_path} not found")
            continue
        print(f"\n=== training: {name} ===")
        log_path = log_dir / f"{name}.log"
        t0 = time.time()
        code = run([sys.executable, "-m", "semdetect.train", "--config", str(config_path)], log_path, job_timeout)
        elapsed_min = (time.time() - t0) / 60
        if code == 0:
            detail = summarize_training_run(config_path)
            status = "ok"
        else:
            detail = f"FAILED exit={code}, see {log_path}"
            status = "failed"
        print(f"  {status} in {elapsed_min:.1f}min - {detail}")
        results.append({"name": name, "status": status, "elapsed_min": elapsed_min, "detail": detail})

    if not args.skip_zero_shot:
        for model in ZERO_SHOT_MODELS:
            print(f"\n=== zero-shot: {model} ===")
            log_path = log_dir / f"zero_shot_{model}.log"
            t0 = time.time()
            code = run(
                [sys.executable, "scripts/zero_shot_eval.py", "--model", model, "--manifest", args.test_manifest],
                log_path,
                job_timeout,
            )
            elapsed_min = (time.time() - t0) / 60
            metrics_path = REPO_ROOT / "outputs" / "zero_shot" / model / "metrics.json"
            if code == 0 and metrics_path.exists():
                import json

                m = json.loads(metrics_path.read_text())
                status, detail = "ok", f"map_50={m.get('map_50', float('nan')):.4f}"
            else:
                status, detail = "failed", f"FAILED exit={code}, see {log_path}"
            print(f"  {status} in {elapsed_min:.1f}min - {detail}")
            results.append({"name": f"zero_shot_{model}", "status": status, "elapsed_min": elapsed_min, "detail": detail})

    if not args.skip_params_report:
        print("\n=== params report ===")
        log_path = log_dir / "params_report.log"
        code = run([sys.executable, "scripts/params_report.py"], log_path, job_timeout)
        status = "ok" if code == 0 else f"FAILED exit={code}"
        results.append({"name": "params_report", "status": status, "elapsed_min": 0.0, "detail": ""})

    total_elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 70}\nsweep summary ({total_elapsed:.1f}min total)\n{'=' * 70}")
    name_width = max(len(r["name"]) for r in results)
    for r in results:
        print(f"  {r['name']:<{name_width}}  {r['status']:<7}  {r['elapsed_min']:>6.1f}min  {r['detail']}")

    n_failed = sum(1 for r in results if r["status"] == "failed")
    if n_failed:
        print(f"\n{n_failed} job(s) failed - check the logs in {log_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()

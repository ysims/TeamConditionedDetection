#!/usr/bin/env python3
"""Runs every experiment in the paper, evaluates it, and writes the LaTeX
tables and result macros that main.tex inputs.

    uv run python scripts/run_experiments.py                  # train + eval + tables
    uv run python scripts/run_experiments.py --stage tables   # regenerate LaTeX only
    uv run python scripts/run_experiments.py --stage tables --placeholder   # "??" everywhere, no results needed

Stages (each skips work that is already done, so an interrupted run resumes):

    train   python -m team_detect.train for every experiment x seed, into
            <output-root>/<experiment>/seed<N>/, logging to train.log there.
            A run whose resolved config differs from the one it was trained
            with is stale: it is refused unless --force retrains it.
    eval    python -m team_detect.evaluation on each run's best.pt -> eval.json,
            plus the colour-rule baselines -> <output-root>/colour_rules.json.
            Re-runs an evaluation whose output lacks a metric this script
            now reports. Latency is measured separately, on an idle machine,
            by scripts/benchmark_latency.py -> <output-root>/latency.json;
            the tables stage uses it if present.
    tables  aggregates mean +- sample std over seeds -> <output-root>/results.json,
            and writes <tables-dir>/*.tex.

Every number in the generated LaTeX comes from eval.json - ground-truth-box
role accuracy and context-swap metrics, and mAP50, all on the test split
with the same fixed teammate/opponent assignment for every seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from team_detect.models.factory import STANDALONE  # noqa: E402
from team_detect.robot_config import RobotConfig, config_to_dict, load_robot_config, save_robot_config  # noqa: E402

# key -> config file. Keys name output dirs; MACRO_STEMS name LaTeX macros.
EXPERIMENTS = {
    "fasterrcnn_film": "configs/fasterrcnn_film.yaml",
    "rtdetr_film": "configs/rtdetr_film.yaml",
    "yolo26n_film": "configs/yolo26n_film.yaml",
    "fasterrcnn_film_distance": "configs/fasterrcnn_film_distance.yaml",
    "fasterrcnn_crossattn": "configs/fasterrcnn_crossattn.yaml",
    "fasterrcnn_crossattn_distance": "configs/fasterrcnn_crossattn_distance.yaml",
    "fasterrcnn_concat": "configs/fasterrcnn_concat.yaml",
    "fasterrcnn_film_shift": "configs/fasterrcnn_film_shift.yaml",
    "standalone_classifier": "configs/standalone_classifier.yaml",
    "rtdetr_film_rw1": "configs/rtdetr_film_rw1.yaml",
    "yolo26n_film_rw1": "configs/yolo26n_film_rw1.yaml",
}
MACRO_STEMS = {
    "fasterrcnn_film": "FrcnnFilm",
    "rtdetr_film": "RtdetrFilm",
    "yolo26n_film": "YoloFilm",
    "fasterrcnn_film_distance": "FrcnnFilmDist",
    "fasterrcnn_crossattn": "FrcnnXattn",
    "fasterrcnn_crossattn_distance": "FrcnnXattnDist",
    "fasterrcnn_concat": "FrcnnConcat",
    "fasterrcnn_film_shift": "FrcnnFilmShift",
    "standalone_classifier": "Standalone",
    "rtdetr_film_rw1": "RtdetrFilmFullWeight",
    "yolo26n_film_rw1": "YoloFilmFullWeight",
}
# eval.json key -> macro suffix. All but role_params are fractions, reported as percentages.
METRICS = {
    "role_acc": "RoleAcc",
    "map_50": "MapFifty",
    "swap_acc": "SwapAcc",
    "both_correct": "BothCorrect",
    "cond_swap_acc": "CondSwapAcc",
    "flip_rate": "FlipRate",
    "teammate_precision": "TeammatePrecision",
    "teammate_recall": "TeammateRecall",
    "opponent_precision": "OpponentPrecision",
    "opponent_recall": "OpponentRecall",
    "role_params": "Params",
}
COLOUR_RULES = {  # colour_rules.json key -> (macro suffix, table label lines)
    "whole_mean": ("RuleWholeMean", ["Mean RGB,", "whole box"]),
    "whole_min": ("RuleWholeMin", ["Min RGB distance,", "whole box"]),
    "centre_mean": ("RuleCentreMean", ["Mean RGB,", "15\\% centre crop"]),
    "centre_min": ("RuleCentreMin", ["Min RGB distance,", "15\\% centre crop"]),
    "torso_hue": ("RuleTorsoHue", ["Mean hue,", "torso band"]),
    "torso_kmeans": ("RuleTorsoKmeans", ["k-means dominant", "colour, torso band"]),
    "calibrated_hue": ("RuleCalibratedHue", ["Calibrated hue,", "torso band"]),
    "calibrated_hsv": ("RuleCalibratedHsv", ["Calibrated HSV,", "torso band"]),
}
CALIBRATED_RULES = {"calibrated_hue": "CalibratedHue", "calibrated_hsv": "CalibratedHsv"}  # -> macro stem of the chosen bin count
# Accuracy by the hue separation of the two supplied colours: bin edges in
# degrees, merged (smallest bin into its smaller neighbour) until every bin
# holds at least SEPARATION_MIN_BOXES test boxes.
SEPARATION_EDGES = [0, 30, 60, 90, 180]
SEPARATION_MIN_BOXES = 40
SEPARATION_RULES = ["torso_hue", "calibrated_hue", "calibrated_hsv"]  # colour_rules.json per_box keys
SEPARATION_MODELS = ["standalone_classifier", "fasterrcnn_film"]
COLOUR_SHIFT = {  # colour_rules.json key -> (macro suffix, LaTeX format of the value)
    "hue_error_median": ("RenderHueErrorMedian", "${:.0f}^\\circ$"),
    "hue_error_p75": ("RenderHueErrorPSeventyFive", "${:.0f}^\\circ$"),
    "saturation_ratio_median": ("RenderSaturationRatio", "{:.2f}"),
    "value_ratio_median": ("RenderValueRatio", "{:.2f}"),
}
LATENCY = {  # latency.json model key -> (macro stem, table label)
    "fasterrcnn": ("Frcnn", "Faster R-CNN"),
    "rtdetr": ("Rtdetr", "RT-DETR"),
    "yolo26n": ("Yolo", "YOLO26n"),
    "standalone": ("Standalone", "Standalone classifier"),
}
ROLE_ONSET_ACC = 0.6  # "role onset": first epoch whose validation role accuracy exceeds this
DEFAULT_SEEDS = [42, 43, 44]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


@dataclass
class Run:
    experiment: str
    seed: int
    dir: Path
    config: RobotConfig

    @property
    def name(self) -> str:
        return f"{self.experiment}/seed{self.seed}"

    @property
    def config_sha(self) -> str:
        return hashlib.sha256(json.dumps(config_to_dict(self.config), sort_keys=True).encode()).hexdigest()[:16]


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def _git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "team_detect", "configs"))}


def resolve_runs(args) -> list[Run]:
    runs = []
    for seed in args.seeds:  # seed-major, so a complete first seed of every table lands early
        for experiment in args.experiments:
            config = load_robot_config(REPO / EXPERIMENTS[experiment])
            run_dir = args.output_root / experiment / f"seed{seed}"
            config.train.seed = seed
            config.train.output_dir = str(run_dir)
            if args.device:
                config.train.device = args.device
            if args.epochs:
                config.train.epochs = args.epochs
            if args.manifest_dir:
                for split in ("train", "val", "test"):
                    setattr(config.data, f"{split}_manifest", str(args.manifest_dir / f"{split}.csv"))
            runs.append(Run(experiment, seed, run_dir, config))
    return runs


# ---------------------------------------------------------------- train


def train(runs: list[Run], args) -> list[str]:
    failures = []
    for i, run in enumerate(runs, 1):
        info = _read_json(run.dir / "run_info.json")
        prefix = f"[train {i}/{len(runs)}] {run.name}"
        if info and info["status"] == "complete" and not args.force:
            if info["config_sha"] != run.config_sha:
                failures.append(f"{run.name}: trained with a different config ({run.dir / 'config.yaml'}); rerun with --force to retrain")
                print(f"{prefix}: STALE - config changed since it was trained")
            else:
                print(f"{prefix}: done, skipping")
            continue
        if run.dir.exists():
            if info is None and any(run.dir.iterdir()):
                raise SystemExit(f"{run.dir} exists but wasn't created by this script (no run_info.json) - refusing to delete it")
            shutil.rmtree(run.dir)  # an incomplete/failed/forced run of ours: start clean
        run.dir.mkdir(parents=True)

        config_path = run.dir / "config.input.yaml"
        save_robot_config(run.config, config_path)
        info = {"status": "running", "config_sha": run.config_sha, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **_git_state()}
        _write_json(run.dir / "run_info.json", info)
        print(f"{prefix}: training -> {run.dir / 'train.log'}", flush=True)

        cmd = [sys.executable, "-m", "team_detect.train", "--config", str(config_path)]
        t0 = time.time()
        with open(run.dir / "train.log", "w") as log:
            try:
                rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_hours * 3600 if args.timeout_hours else None).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        ok = rc == 0 and (run.dir / "metrics" / "final.json").exists()
        info.update(status="complete" if ok else "failed", returncode=rc, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), hours=round((time.time() - t0) / 3600, 2))
        _write_json(run.dir / "run_info.json", info)
        print(f"{prefix}: {info['status']} in {info['hours']}h")
        if not ok:
            failures.append(f"{run.name}: training {info['status']} (returncode {rc}), see {run.dir / 'train.log'}")
    return failures


# ---------------------------------------------------------------- eval


def _checkpoint_id(run: Run) -> str:
    st = (run.dir / "checkpoints" / "best.pt").stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def evaluate(runs: list[Run], args) -> list[str]:
    failures = []
    for i, run in enumerate(runs, 1):
        prefix = f"[eval {i}/{len(runs)}] {run.name}"
        info = _read_json(run.dir / "run_info.json")
        if not info or info["status"] != "complete":
            print(f"{prefix}: not trained, skipping")
            continue
        existing = _read_json(run.dir / "eval.json")
        expected = {*METRICS, "per_box_correct"} - ({"map_50"} if run.config.model.architecture == STANDALONE else set())
        if existing and existing.get("checkpoint") == _checkpoint_id(run) and expected <= existing.keys() and not args.reeval:
            print(f"{prefix}: done, skipping")
            continue
        print(f"{prefix}: evaluating", flush=True)
        out = run.dir / "eval.json"
        cmd = [sys.executable, "-m", "team_detect.evaluation", "--run-dir", str(run.dir), "--device", run.config.train.device, "--output", str(out)]
        with open(run.dir / "eval.log", "w") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            failures.append(f"{run.name}: evaluation failed, see {run.dir / 'eval.log'}")
            continue
        _write_json(out, {**_read_json(out), "checkpoint": _checkpoint_id(run), "config_sha": info["config_sha"]})

    rules_path = args.output_root / "colour_rules.json"
    rules = _read_json(rules_path)
    if rules is None or not {*COLOUR_RULES, *COLOUR_SHIFT, "per_box"} <= rules.keys() or args.reeval:
        data = runs[0].config.data
        print("[eval] colour-rule baselines", flush=True)
        args.output_root.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "team_detect.evaluation", "--colour-rules", "--manifest", data.test_manifest,
               "--train-manifest", data.train_manifest, "--val-manifest", data.val_manifest,
               "--img-size", str(data.img_size), "--eval-seed", str(data.eval_seed), "--output", str(rules_path)]
        if subprocess.run(cmd, cwd=REPO, capture_output=True).returncode != 0:
            failures.append("colour-rule baselines: evaluation failed")
    return failures


# ---------------------------------------------------------------- tables


def role_onset_epoch(run: Run) -> int | None:
    """First epoch whose validation role accuracy exceeds ROLE_ONSET_ACC -
    how long a role head takes to start using the context at all."""
    with open(run.dir / "metrics" / "history.csv") as f:
        for row in csv.DictReader(f):
            if row.get("val_role_acc") and float(row["val_role_acc"]) > ROLE_ONSET_ACC:
                return int(row["epoch"])
    return None


def _seed_stat(values: list[float]) -> dict:
    return {"values": values, "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None}


def separation_bins(delta_hue: list[float]) -> list[tuple[float, float, list[int]]]:
    """(lo, hi, box indices) per SEPARATION_EDGES bin, after merging small bins.
    The last bin includes its upper edge."""
    bins = [[lo, hi, [i for i, d in enumerate(delta_hue) if lo <= d < hi or (hi == SEPARATION_EDGES[-1] and d == hi)]]
            for lo, hi in zip(SEPARATION_EDGES, SEPARATION_EDGES[1:])]
    while len(bins) > 1 and min(len(b[2]) for b in bins) < SEPARATION_MIN_BOXES:
        i = min(range(len(bins)), key=lambda j: len(bins[j][2]))
        j = i - 1 if i == len(bins) - 1 or (i > 0 and len(bins[i - 1][2]) <= len(bins[i + 1][2])) else i + 1
        a, b = sorted((i, j))
        bins[a:b + 1] = [[bins[a][0], bins[b][1], sorted(bins[a][2] + bins[b][2])]]
    return [tuple(b) for b in bins]


def separation(rules: dict, per_box: dict[str, list[list[int]]]) -> list[dict]:
    """Accuracy per hue-separation bin: the colour rules', and mean +- std
    over seeds of each SEPARATION_MODELS experiment's (per_box: exp -> one
    per-box correctness list per seed)."""
    out = []
    for lo, hi, idx in separation_bins(rules["per_box"]["delta_hue"]):
        row = {"range": [lo, hi], "boxes": len(idx)}
        for rule in SEPARATION_RULES:
            row[rule] = statistics.mean(rules["per_box"][rule][i] for i in idx)
        for exp, seeds in per_box.items():
            row[exp] = _seed_stat([statistics.mean(correct[i] for i in idx) for correct in seeds])
        out.append(row)
    return out


def aggregate(runs: list[Run], args) -> dict:
    """{"experiments": {exp: {metric: {"values", "mean", "std", "seeds"}}}, "colour_rules": {...}, ...}"""
    per_exp: dict[str, dict[str, list]] = {e: {} for e in args.experiments}
    per_box: dict[str, list[list[int]]] = {}
    missing, stale = [], []
    onsets: dict[str, list[int | None]] = {}
    for run in runs:
        result = _read_json(run.dir / "eval.json")
        if result is None:
            missing.append(run.name)
            continue
        if result["config_sha"] != run.config_sha:
            stale.append(run.name)
            continue
        for metric in METRICS:
            if metric in result:
                per_exp[run.experiment].setdefault(metric, []).append((run.seed, result[metric]))
        if run.experiment in SEPARATION_MODELS and "per_box_correct" in result:
            per_box.setdefault(run.experiment, []).append(result["per_box_correct"])
        onsets.setdefault(run.experiment, []).append(role_onset_epoch(run))
    if stale:
        raise SystemExit("Stale results (config changed since training; retrain with --stage train --force):\n  " + "\n  ".join(stale))
    if missing and not args.allow_partial:
        raise SystemExit("Missing results (run the train and eval stages, or pass --allow-partial):\n  " + "\n  ".join(missing))

    experiments = {}
    for exp, metrics in per_exp.items():
        experiments[exp] = {}
        for metric, seeded in metrics.items():
            experiments[exp][metric] = {"seeds": [s for s, _ in seeded], **_seed_stat([v for _, v in seeded])}
    rules = _read_json(args.output_root / "colour_rules.json")
    if rules is None and not args.allow_partial:
        raise SystemExit("Missing colour_rules.json (run the eval stage, or pass --allow-partial)")
    n_seeds = {len(m["values"]) for e in experiments.values() for m in e.values()}
    return {
        "experiments": experiments,
        "colour_rules": {k: v for k, v in (rules or {}).items() if k != "per_box"},
        "separation": separation(rules, per_box) if rules and "per_box" in rules else [],
        "latency": _read_json(args.output_root / "latency.json") or {},
        "role_onset_epochs": onsets,
        "n_boxes": (rules or {}).get("n_boxes"),
        "n_seeds": n_seeds.pop() if len(n_seeds) == 1 else None,  # None: experiments have unequal seed counts
        "missing": missing,
    }


class Formatter:
    """Formats aggregated values as LaTeX; placeholder=True renders '??'."""

    def __init__(self, results: dict | None):
        self.results = results

    def stat(self, experiment: str, metric: str) -> dict | None:
        if self.results is None:
            return None
        return self.results["experiments"].get(experiment, {}).get(metric)

    def value(self, experiment: str, metric: str, percent_sign: bool = False, bold: bool = False) -> str:
        """Math-mode '$92.1 \\pm 1.5$' (accuracies are stored as fractions)."""
        s = self.stat(experiment, metric)
        if metric == "role_params":
            text = "??" if s is None else f"{round(s['mean']):,}"
            return f"$\\mathbf{{{text}}}$" if bold else text
        if s is None:
            body = "??"
        else:
            body = f"{100 * s['mean']:.1f}" + ("" if s["std"] is None else f" \\pm {100 * s['std']:.1f}")
        body += "\\%" if percent_sign else ""
        return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"

    def rule(self, key: str, percent_sign: bool = False, bold: bool = False) -> str:
        rules = self.results["colour_rules"] if self.results else {}
        body = "??" if key not in rules else f"{100 * rules[key]:.1f}"
        body += "\\%" if percent_sign else ""
        return f"$\\mathbf{{{body}}}$" if bold else (f"${body}$" if percent_sign else body)

    def colour_shift(self, key: str) -> str:
        rules = self.results["colour_rules"] if self.results else {}
        template = COLOUR_SHIFT[key][1]
        return template.format(rules[key]) if key in rules else template.replace("{:.0f}", "??").replace("{:.2f}", "??")

    def latency(self, device: str, model: str, part: str) -> str:
        """part: role_head, standalone or detector. Milliseconds: 2 decimals below 1 ms, 1 below 100 ms."""
        value = ((self.results or {}).get("latency", {}).get(device, {}).get(model, {})).get(f"{part}_ms")
        if value is None:
            return "??"
        return f"{value:.2f}" if value < 1 else f"{value:.1f}" if value < 100 else f"{value:.0f}"

    def role_onset(self, experiment: str) -> str:
        """Range over seeds, e.g. '62--114'; 'never' if some seed never got there."""
        epochs = ((self.results or {}).get("role_onset_epochs") or {}).get(experiment)
        if not epochs:
            return "??"
        if None in epochs:
            return "never"
        lo, hi = min(epochs), max(epochs)
        return str(lo) if lo == hi else f"{lo}--{hi}"

    def latency_hardware(self, key: str) -> str:
        return (self.results or {}).get("latency", {}).get(key, "??")

    def rules_shown(self) -> list[str]:
        """COLOUR_RULES keys in Table 3: calibrated_hsv (the optional
        extension) only if its validation accuracy beats calibrated_hue's."""
        rules = self.results["colour_rules"] if self.results else {}
        hsv_val, hue_val = rules.get("calibrated_hsv_val_acc"), rules.get("calibrated_hue_val_acc")
        keep_hsv = hsv_val is None or hue_val is None or max(hsv_val.values()) > max(hue_val.values())
        return [k for k in COLOUR_RULES if k != "calibrated_hsv" or keep_hsv]

    def calibration_bins(self, rule: str) -> str:
        return str((self.results or {}).get("colour_rules", {}).get(f"{rule}_bins", "??"))

    def separation_rows(self) -> list[dict]:
        """Per bin; placeholder bins (no values) when there are no results."""
        rows = (self.results or {}).get("separation")
        return rows or [{"range": [lo, hi]} for lo, hi in zip(SEPARATION_EDGES, SEPARATION_EDGES[1:])]

    def separation(self, row: dict, key: str, percent_sign: bool = False) -> str:
        """key: 'range', 'boxes', a SEPARATION_RULES key or a SEPARATION_MODELS experiment."""
        if key == "range":
            lo, hi = row["range"]
            return f"${lo}$--${hi}^\\circ$"
        pct = "\\%" if percent_sign else ""
        if key not in row:
            return f"$??{pct}$" if key != "boxes" else "??"
        if key == "boxes":
            return str(row["boxes"])
        value = row[key]
        if isinstance(value, dict):
            body = f"{100 * value['mean']:.1f}" + ("" if value["std"] is None else f" \\pm {100 * value['std']:.1f}")
        else:
            body = f"{100 * value:.1f}"
        return f"${body}{pct}$"

    def best(self, experiments: list[str], metric: str, lowest: bool = False) -> str | None:
        means = {e: s["mean"] for e in experiments if (s := self.stat(e, metric))}
        if not means:
            return None
        return (min if lowest else max)(means, key=means.get)


def _rows(label_lines: list[str], cells: list[str]) -> list[str]:
    """First label line carries the cells; continuation lines get empty cells,
    matching the paper's narrow-column style."""
    lines = [" & ".join([label_lines[0], *cells]) + " \\\\"]
    lines += [" & ".join([extra, *[""] * len(cells)]) + " \\\\" for extra in label_lines[1:]]
    return lines


def _tabular(spec: str, header: list[str], body: list[str]) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", " & ".join(header) + " \\\\", "\\hline", *body, "\\end{tabular}", ""])


def render_tables(fmt: Formatter) -> dict[str, str]:
    acc, map50 = "\\textbf{Role Acc. (\\%)}", "\\textbf{$\\mathrm{mAP}_{50}$ (\\%)}"

    def bold_table(rows: list[tuple[list[str], str]]) -> list[str]:
        exps = [e for _, e in rows]
        best = {m: fmt.best(exps, m) for m in ("role_acc", "map_50")}
        return [line for label, e in rows for line in _rows(label, [fmt.value(e, m, bold=best[m] == e) for m in ("role_acc", "map_50")])]

    detectors = bold_table([(["Faster R-CNN"], "fasterrcnn_film"), (["RT-DETR"], "rtdetr_film"), (["YOLO26n"], "yolo26n_film")])
    conditioning = bold_table([
        (["FiLM (RGB)"], "fasterrcnn_film"),
        (["Shift-only FiLM"], "fasterrcnn_film_shift"),
        (["Concatenation", "+ MLP"], "fasterrcnn_concat"),
        (["FiLM (RGB)", "+ distance"], "fasterrcnn_film_distance"),
        (["Cross-attention"], "fasterrcnn_crossattn"),
        (["Cross-attention", "+ distance"], "fasterrcnn_crossattn_distance"),
    ])

    learned = [(["Standalone classifier"], "standalone_classifier"), (["FiLM, integrated"], "fasterrcnn_film")]
    learned_exps = [e for _, e in learned]
    best_acc, fewest = fmt.best(learned_exps, "role_acc"), fmt.best(learned_exps, "role_params", lowest=True)
    role_rows = [line for key in fmt.rules_shown() for line in _rows(COLOUR_RULES[key][1], [fmt.rule(key), "--"])]
    role_rows.append("\\hline")
    role_rows += [line for label, e in learned for line in _rows(label, [fmt.value(e, "role_acc", bold=e == best_acc), fmt.value(e, "role_params", bold=e == fewest)])]

    swap_metrics = [
        (["Original-context acc"], "role_acc"),
        (["Swapped-context acc"], "swap_acc"),
        (["Both correct"], "both_correct"),
        (["$P(\\mathrm{swap\\ correct}\\mid$", "$\\mathrm{original\\ correct})$"], "cond_swap_acc"),
        (["Flip rate"], "flip_rate"),
    ]
    swap_rows = [line for label, m in swap_metrics for line in _rows(label, [fmt.value("yolo26n_film", m), fmt.value("fasterrcnn_film", m)])]

    per_class_metrics = [
        (["Teammate precision"], "teammate_precision"),
        (["Teammate recall"], "teammate_recall"),
        (["Opponent precision"], "opponent_precision"),
        (["Opponent recall"], "opponent_recall"),
    ]
    per_class_rows = [line for label, m in per_class_metrics for line in _rows(label, [fmt.value("yolo26n_film", m), fmt.value("fasterrcnn_film", m)])]

    latency_rows = [line for model in ("fasterrcnn", "rtdetr", "yolo26n") for line in _rows([f"Role head, {LATENCY[model][1]}"], [fmt.latency(d, model, "role_head") for d in ("cuda", "cpu")])]
    latency_rows += _rows(["Standalone classifier"], [fmt.latency(d, "standalone", "standalone") for d in ("cuda", "cpu")])
    latency_rows.append("\\hline")
    latency_rows += [line for model in ("fasterrcnn", "rtdetr", "yolo26n") for line in _rows([f"Detector, {LATENCY[model][1]}"], [fmt.latency(d, model, "detector") for d in ("cuda", "cpu")])]

    sep_bins = fmt.separation_rows()
    sep_methods = [(["Mean hue, torso band"], "torso_hue"), (["Calibrated hue"], "calibrated_hue"), (["Calibrated HSV"], "calibrated_hsv"),
                   (["Standalone classifier"], "standalone_classifier"), (["FiLM, integrated"], "fasterrcnn_film")]
    sep_methods = [(label, key) for label, key in sep_methods if key in SEPARATION_MODELS or key in fmt.rules_shown()]
    sep_rows = _rows(["Boxes"], [fmt.separation(b, "boxes") for b in sep_bins]) + ["\\hline"]
    sep_rows += [line for label, key in sep_methods for line in _rows(label, [fmt.separation(b, key) for b in sep_bins])]

    return {
        "table_separation.tex": _tabular("l|" + "c" * len(sep_bins), ["\\textbf{$\\Delta h$}", *[fmt.separation(b, "range") for b in sep_bins]], sep_rows),
        "table_detectors.tex": _tabular("l|cc", ["\\textbf{Detector}", acc, map50], detectors),
        "table_conditioning.tex": _tabular("l|cc", ["\\textbf{Role Head}", acc, map50], conditioning),
        "table_role_classification.tex": _tabular("l|c|c", ["\\textbf{Method}", acc, "\\textbf{Parameters}"], role_rows),
        "table_swap.tex": _tabular("l|c|c", ["\\textbf{Metric}", "\\textbf{YOLO26n}", "\\small{\\textbf{Faster R-CNN}}"], swap_rows),
        "table_per_class.tex": _tabular("l|c|c", ["\\textbf{Metric}", "\\textbf{YOLO26n}", "\\small{\\textbf{Faster R-CNN}}"], per_class_rows),
        "table_latency.tex": _tabular("l|c|c", ["\\textbf{Component}", "\\textbf{GPU (ms)}", "\\textbf{CPU (ms)}"], latency_rows),
    }


def render_macros(fmt: Formatter, results: dict | None) -> str:
    """Prose macros, e.g. \\resFrcnnFilmRoleAcc -> $92.1 \\pm 1.5\\%$."""
    lines = ["% Generated by scripts/run_experiments.py - do not edit.", ""]
    n_seeds = results["n_seeds"] if results else None
    lines.append(f"\\newcommand{{\\resNumSeeds}}{{{NUMBER_WORDS.get(n_seeds, '??')}}}")
    lines.append(f"\\newcommand{{\\resTestBoxes}}{{{(results or {}).get('n_boxes') or '??'}}}")
    for exp, stem in MACRO_STEMS.items():
        for metric, suffix in METRICS.items():
            lines.append(f"\\newcommand{{\\res{stem}{suffix}}}{{{fmt.value(exp, metric, percent_sign=True)}}}")
    for key, (suffix, _) in COLOUR_RULES.items():
        lines.append(f"\\newcommand{{\\res{suffix}}}{{{fmt.rule(key, percent_sign=True)}}}")
    for exp, stem in MACRO_STEMS.items():
        lines.append(f"\\newcommand{{\\res{stem}RoleOnset}}{{{fmt.role_onset(exp)}}}")
    lines.append(f"\\newcommand{{\\resRoleOnsetAcc}}{{${100 * ROLE_ONSET_ACC:.0f}\\%$}}")
    for key, (suffix, _) in COLOUR_SHIFT.items():
        lines.append(f"\\newcommand{{\\res{suffix}}}{{{fmt.colour_shift(key)}}}")
    for rule, stem in CALIBRATED_RULES.items():
        lines.append(f"\\newcommand{{\\res{stem}Bins}}{{{fmt.calibration_bins(rule)}}}")
    # \resSepOneRange, \resSepOneBoxes, \resSepOneTorsoHue, ..., \resSepOneFrcnnFilm - bins in increasing separation
    sep_stems = {"range": "Range", "boxes": "Boxes", "torso_hue": "TorsoHue", "calibrated_hue": "CalibratedHue", "calibrated_hsv": "CalibratedHsv",
                 **{exp: MACRO_STEMS[exp] for exp in SEPARATION_MODELS}}
    sep_bins = fmt.separation_rows()
    lines.append(f"\\newcommand{{\\resSepNumBins}}{{{NUMBER_WORDS[len(sep_bins)]}}}")
    for i, row in enumerate(sep_bins, 1):
        for key, stem in sep_stems.items():
            lines.append(f"\\newcommand{{\\resSep{NUMBER_WORDS[i].capitalize()}{stem}}}{{{fmt.separation(row, key, percent_sign=True)}}}")
    for model, (stem, _) in LATENCY.items():
        for part in ("standalone",) if model == "standalone" else ("role_head", "detector"):
            part_name = {"role_head": "RoleHead", "detector": "Detector", "standalone": ""}[part]
            for device, device_name in (("cuda", "Gpu"), ("cpu", "Cpu")):
                lines.append(f"\\newcommand{{\\resLatency{stem}{part_name}{device_name}}}{{{fmt.latency(device, model, part)}}}")
    lines.append(f"\\newcommand{{\\resLatencyGpuName}}{{{fmt.latency_hardware('gpu_name')}}}")
    lines.append(f"\\newcommand{{\\resLatencyCpuName}}{{{fmt.latency_hardware('cpu_name')}}}")
    return "\n".join(lines) + "\n"


def write_tables(runs: list[Run], args) -> None:
    results = None
    if not args.placeholder:
        results = aggregate(runs, args)
        _write_json(args.output_root / "results.json", results)
        if results["missing"]:
            print(f"[tables] WARNING: partial results, missing {len(results['missing'])} run(s): {', '.join(results['missing'])}")
    fmt = Formatter(results)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    files = {**render_tables(fmt), "results_macros.tex": render_macros(fmt, results)}
    header = "% Generated by scripts/run_experiments.py - do not edit.\n"
    for name, text in files.items():
        (args.tables_dir / name).write_text(text if text.startswith("%") else header + text)
    print(f"[tables] wrote {', '.join(sorted(files))} to {args.tables_dir}")


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "train", "eval", "tables"], default="all")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs" / "paper")
    parser.add_argument("--tables-dir", type=Path, default=REPO / "tables")
    parser.add_argument("--device", help="Override every config's train.device")
    parser.add_argument("--force", action="store_true", help="Retrain runs even if complete (including stale ones)")
    parser.add_argument("--reeval", action="store_true", help="Re-run evaluation even if eval.json is current")
    parser.add_argument("--timeout-hours", type=float, default=12.0, help="Kill a training run after this long (0 = no limit)")
    parser.add_argument("--allow-partial", action="store_true", help="Write tables from whatever results exist")
    parser.add_argument("--placeholder", action="store_true", help="Write tables with ?? for every value (tables stage only)")
    parser.add_argument("--epochs", type=int, help="Smoke testing: override train.epochs")
    parser.add_argument("--manifest-dir", type=Path, help="Smoke testing: use train/val/test.csv from this dir")
    args = parser.parse_args()
    args.output_root, args.tables_dir = args.output_root.resolve(), args.tables_dir.resolve()

    runs = resolve_runs(args)
    failures = []
    if args.stage in ("all", "train"):
        failures += train(runs, args)
    if args.stage in ("all", "eval"):
        failures += evaluate(runs, args)
    if args.stage in ("all", "tables"):
        if failures and not args.allow_partial:
            print("\nNot writing tables - failures above.")
        else:
            write_tables(runs, args)
    if failures:
        print("\nFailures:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()

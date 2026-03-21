#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from sklearn.datasets import get_data_home


# From the public Hugging Face config for Qwen/Qwen3-VL-8B-Instruct:
#   vision_config.depth = 27
#   text_config.num_hidden_layers = 36
QWEN3_VL_8B_VISUAL_DEPTH = 27
QWEN3_VL_8B_LANGUAGE_DEPTH = 36

# Image-based NSD benchmarks (visual pathway; NSD images as stimuli)
# Restricted to a diverse set of six ROIs:
# V4, ventral stream, parietal, lateral, high-level ventral, high-level lateral.
DEFAULT_VISUAL_BENCHMARKS = [
    "NSDV4Shared",
    "NSDVentralShared",
    "NSDParietalShared",
    "NSDLateralShared",
    "NSDHighVentralShared",
    "NSDHighLateralShared",
]

# Caption-based NSD benchmarks (language pathway; NSD captions as stimuli)
# Caption variants of the same six ROIs.
DEFAULT_CAPTION_BENCHMARKS = [
    "NSDV4CaptionShared",
    "NSDVentralCaptionShared",
    "NSDParietalCaptionShared",
    "NSDLateralCaptionShared",
    "NSDHighVentralCaptionShared",
    "NSDHighLateralCaptionShared",
]


def load_sweep_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Sweep config not found: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def discover_qwen_layers(model_identifier: str) -> dict:
    if model_identifier not in {"qwen3_vl_8b_img", "qwen3_vl_8b_txt", "qwen3_vl_8b_joint"}:
        raise ValueError(
            f"Unsupported model for this sweep helper: {model_identifier}. "
            "Expected one of ['qwen3_vl_8b_img', 'qwen3_vl_8b_joint', 'qwen3_vl_8b_txt']."
        )

    return {
        "visual_indices": list(range(QWEN3_VL_8B_VISUAL_DEPTH)),
        "language_indices": list(range(QWEN3_VL_8B_LANGUAGE_DEPTH)),
    }


def select_indices(indices: list[int], step: int) -> list[int]:
    if not indices:
        return []

    selected = indices[::step]
    if indices[-1] not in selected:
        selected.append(indices[-1])
    return selected


def ensure_output_root(output_root: str | None) -> Path:
    if output_root:
        path = Path(output_root).expanduser().resolve()
    elif "RESULTS_PATH" in os.environ:
        path = Path(os.environ["RESULTS_PATH"]).expanduser().resolve()
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = Path(get_data_home()).resolve() / f"qwen_visual_sweep_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_run_log(path: Path, payload: list[dict]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def summarize_results(
    output_root: Path,
    model: str,
    benchmarks: list[str],
    metric: str,
    selected_layers: list[str],
) -> tuple[Path, Path]:
    results_dir = output_root / "results"
    summary_rows = []

    for benchmark in benchmarks:
        for layer_name in selected_layers:
            row = {
                "benchmark": benchmark,
                "layer": layer_name,
                "status": "missing",
                "final_pearson": None,
                "final_unceiled_pearson": None,
                "final_r2": None,
                "final_unceiled_r2": None,
                "timestamp": None,
                "results_file": str(results_dir / f"{model}_{layer_name}_{benchmark}.pkl"),
            }

            result_file = Path(row["results_file"])
            if not result_file.exists():
                summary_rows.append(row)
                continue

            with result_file.open("rb") as handle:
                payload = pickle.load(handle)

            metrics_block = payload.get("metrics", {})
            # Re-runs append to a list of metric dicts; use the latest run for summary.
            if isinstance(metrics_block, list):
                if not metrics_block:
                    metrics_block = {}
                else:
                    metrics_block = metrics_block[-1]
            if not isinstance(metrics_block, dict):
                metrics_block = {}
            metric_block = metrics_block.get(metric)
            if metric_block is None:
                row["status"] = "missing_metric"
                row["timestamp"] = metrics_block.get("timestamp")
                summary_rows.append(row)
                continue

            row["status"] = "ok"
            row["timestamp"] = metrics_block.get("timestamp")
            row["final_pearson"] = metric_block.get("final_pearson")
            row["final_unceiled_pearson"] = metric_block.get("final_unceiled_pearson")
            row["final_r2"] = metric_block.get("final_r2")
            row["final_unceiled_r2"] = metric_block.get("final_unceiled_r2")
            summary_rows.append(row)

    csv_path = output_root / "qwen_visual_sweep_summary.csv"
    json_path = output_root / "qwen_visual_sweep_summary.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
                "layer",
                "status",
                "final_pearson",
                "final_unceiled_pearson",
                "final_r2",
                "final_unceiled_r2",
                "timestamp",
                "results_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with json_path.open("w") as handle:
        json.dump(summary_rows, handle, indent=2)

    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep Qwen3-VL layers for BBScore in different modes (visual or language, image or caption)."
    )
    parser.add_argument("--model", default="qwen3_vl_8b_img")
    parser.add_argument(
        "--config",
        default="qwen_sweep_config.json",
        help="Path to JSON config controlling default benchmarks and layer indices per sweep mode.",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=[
            "visual_image",
            "language_image",
            "language_caption",
            "language_caption_long",
            "language_multimodal",
        ],
        default="language_image",
        help=(
            "Which pathway/input to sweep: "
            "'visual_image' (vision blocks, image input), "
            "'language_image' (language blocks, image+prompt input), "
            "'language_caption' / 'language_caption_long' (language blocks, caption-only; long uses "
            "a separate config block—set NSD_CAPTIONS_PATH to your JSONL), "
            "'language_multimodal' (single forward: image + caption from NSD_CAPTIONS_PATH, language layers, qwen3_vl_8b_joint; "
            "batching uses the same per-sample preprocess + collate path as other Qwen modes)."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--layer-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit layer indices to override the config for the selected sweep mode.",
    )
    parser.add_argument("--metric", default="torch_ridge")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--use-ridge-smart-memory", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    if args.sweep_mode not in config:
        raise KeyError(
            f"Sweep mode '{args.sweep_mode}' not found in config {Path(args.config).resolve()}"
        )
    mode_config = config[args.sweep_mode]

    # Configure model + default benchmarks based on sweep_mode, unless
    # the user explicitly overrides benchmarks on the CLI.
    if args.sweep_mode == "visual_image":
        if args.model != "qwen3_vl_8b_img":
            args.model = "qwen3_vl_8b_img"
    elif args.sweep_mode == "language_image":
        if args.model != "qwen3_vl_8b_img":
            args.model = "qwen3_vl_8b_img"
    elif args.sweep_mode in ("language_caption", "language_caption_long"):
        if args.model != "qwen3_vl_8b_txt":
            args.model = "qwen3_vl_8b_txt"
    elif args.sweep_mode == "language_multimodal":
        if args.model != "qwen3_vl_8b_joint":
            args.model = "qwen3_vl_8b_joint"

    if args.benchmarks is None:
        args.benchmarks = mode_config.get("benchmarks", [])

    output_root = ensure_output_root(args.output_root)
    discovery = discover_qwen_layers(args.model)

    visual_indices = discovery["visual_indices"]
    language_indices = discovery["language_indices"]

    if not visual_indices:
        raise RuntimeError("No visual block layers matching 'model.visual.blocks.<idx>' were found.")

    if args.layer_indices is not None:
        selected_indices = args.layer_indices
    else:
        selected_indices = mode_config.get("layer_indices")

    if not selected_indices:
        # Fallback to historical step-based behavior if no config list is supplied.
        if args.sweep_mode == "visual_image":
            selected_indices = select_indices(visual_indices, args.step)
        else:
            selected_indices = select_indices(language_indices, args.step) if language_indices else []

    selected_visual = selected_indices if args.sweep_mode == "visual_image" else select_indices(visual_indices, args.step)
    selected_language = selected_indices if args.sweep_mode != "visual_image" else []

    if args.sweep_mode == "visual_image":
        selected_layers = [f"model.visual.blocks.{idx}" for idx in selected_indices]
    else:
        selected_layers = [f"model.language_model.layers.{idx}" for idx in selected_indices]

    print(f"Discovered {len(visual_indices)} visual blocks: {visual_indices[0]}..{visual_indices[-1]}")
    if language_indices:
        print(f"Discovered {len(language_indices)} language blocks: {language_indices[0]}..{language_indices[-1]}")
    else:
        print("Did not detect language block names with the current heuristics.")
    if args.sweep_mode == "visual_image":
        print(f"[Mode: visual_image] Selected visual blocks: {selected_indices}")
    else:
        print(f"[Mode: {args.sweep_mode}] Selected language blocks: {selected_indices}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Using sweep config: {Path(args.config).expanduser().resolve()}")

    base_command = [
        sys.executable,
        "run.py",
        "--model",
        args.model,
        "--metric",
        args.metric,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.use_ridge_smart_memory:
        base_command.append("--use-ridge-smart-memory")

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "benchmarks": args.benchmarks,
        "metric": args.metric,
        "batch_size": args.batch_size,
        "step": args.step,
        "sweep_mode": args.sweep_mode,
        "use_ridge_smart_memory": args.use_ridge_smart_memory,
        "continue_on_error": args.continue_on_error,
        "output_root": str(output_root),
        "config_path": str(Path(args.config).expanduser().resolve()),
        "visual_indices": visual_indices,
        "language_indices": language_indices,
        "selected_visual_indices": selected_visual,
        "selected_language_indices": selected_language,
        "selected_layers": selected_layers,
        # commands will depend on sweep_mode; populated below
        "commands": [],
    }

    commands: list[list[str]] = []
    if args.sweep_mode == "visual_image":
        # Visual sweeps: run one run.py call per layer per benchmark.
        for benchmark in args.benchmarks:
            for layer_name in selected_layers:
                cmd = base_command + ["--layer", layer_name, "--benchmark", benchmark]
                commands.append(cmd)
    else:
        # Language sweeps: run one run.py call per benchmark with all layers at once.
        if not selected_layers:
            raise RuntimeError("No language layers selected for language sweep.")
        for benchmark in args.benchmarks:
            cmd = base_command + ["--layer", *selected_layers, "--benchmark", benchmark]
            commands.append(cmd)

    manifest["commands"] = commands
    write_manifest(output_root / "qwen_visual_sweep_manifest.json", manifest)

    if args.dry_run:
        print("Dry run commands:")
        for cmd in commands:
            print(" ".join(cmd))
        return 0

    env = os.environ.copy()
    env["RESULTS_PATH"] = str(output_root)
    run_log_path = output_root / "qwen_visual_sweep_run_log.json"
    run_log: list[dict] = []
    failed_benchmarks: list[str] = []

    print(f"Writing raw results under: {output_root / 'results'}")
    for cmd in commands:
        print("Running command:")
        print(" ".join(cmd))
        started_utc = datetime.now(UTC).isoformat()
        try:
            subprocess.run(cmd, check=True, env=env)
            status = "ok"
            return_code = 0
        except subprocess.CalledProcessError as exc:
            status = "failed"
            return_code = exc.returncode
            # Benchmark name is always the last argument after '--benchmark'
            benchmark_name = cmd[-1] if cmd[-2] == "--benchmark" else "unknown"
            failed_benchmarks.append(benchmark_name)
            print(
                f"Command failed with return code {return_code}. "
                f"See the Slurm log above for the exact failing command output."
            )
            if not args.continue_on_error:
                run_log.append(
                    {
                        "benchmark": benchmark_name,
                        "status": status,
                        "return_code": return_code,
                        "started_utc": started_utc,
                        "finished_utc": datetime.now(UTC).isoformat(),
                        "command": cmd,
                    }
                )
                write_run_log(run_log_path, run_log)
                break
        finished_utc = datetime.now(UTC).isoformat()
        run_log.append(
            {
                "benchmark": cmd[-1] if len(cmd) >= 2 and cmd[-2] == "--benchmark" else "unknown",
                "status": status,
                "return_code": return_code,
                "started_utc": started_utc,
                "finished_utc": finished_utc,
                "command": cmd,
            }
        )
        write_run_log(run_log_path, run_log)

    csv_path, json_path = summarize_results(
        output_root=output_root,
        model=args.model,
        benchmarks=args.benchmarks,
        metric=args.metric,
        selected_layers=selected_layers,
    )
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")
    if failed_benchmarks:
        print(f"Benchmarks with failures: {', '.join(failed_benchmarks)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

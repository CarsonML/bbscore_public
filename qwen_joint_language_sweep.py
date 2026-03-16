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

from qwen_visual_sweep import ensure_output_root, load_sweep_config, write_manifest, write_run_log


def _normalize_caption_benchmark(benchmark: str) -> str:
    if benchmark.endswith("CaptionShared"):
        return benchmark.replace("CaptionShared", "Shared")
    return benchmark


def _joint_layer_name(layer_name: str) -> str:
    return (
        "Joint_"
        f"qwen3_vl_8b_img:{layer_name}_"
        f"qwen3_vl_8b_txt:{layer_name}"
    )


def summarize_results(
    output_root: Path,
    benchmarks: list[str],
    selected_layers: list[str],
    metric: str,
) -> tuple[Path, Path]:
    results_dir = output_root / "results"
    model_prefix = "Joint_qwen3_vl_8b_img_qwen3_vl_8b_txt"
    summary_rows = []

    for benchmark in benchmarks:
        for layer_name in selected_layers:
            joint_layer_name = _joint_layer_name(layer_name)
            result_file = results_dir / f"{model_prefix}_{joint_layer_name}_{benchmark}.pkl"
            row = {
                "benchmark": benchmark,
                "layer": layer_name,
                "status": "missing",
                "final_pearson": None,
                "final_unceiled_pearson": None,
                "final_r2": None,
                "final_unceiled_r2": None,
                "timestamp": None,
                "results_file": str(result_file),
            }

            if not result_file.exists():
                summary_rows.append(row)
                continue

            with result_file.open("rb") as handle:
                payload = pickle.load(handle)

            metrics_block = payload.get("metrics", {})
            if isinstance(metrics_block, list) and metrics_block:
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

    csv_path = output_root / "qwen_joint_language_sweep_summary.csv"
    json_path = output_root / "qwen_joint_language_sweep_summary.json"

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
        description=(
            "Sweep joint Qwen language layers by concatenating image-input and "
            "caption-input language features for aligned NSD benchmarks."
        )
    )
    parser.add_argument(
        "--config",
        default="qwen_sweep_config.json",
        help="Path to JSON config controlling NSD benchmark and layer defaults.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional override for the NSD shared benchmarks to run.",
    )
    parser.add_argument(
        "--layer-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit language layer indices to override the config defaults.",
    )
    parser.add_argument("--metric", default="joint_ridge")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_sweep_config(args.config)
    image_cfg = config["language_image"]
    caption_cfg = config["language_caption"]

    image_benchmarks = image_cfg.get("benchmarks", [])
    caption_benchmarks = caption_cfg.get("benchmarks", [])
    normalized_caption = [_normalize_caption_benchmark(b) for b in caption_benchmarks]
    if image_benchmarks != normalized_caption:
        raise ValueError(
            "language_image and language_caption benchmarks are not aligned in qwen_sweep_config.json."
        )

    image_layers = image_cfg.get("layer_indices", [])
    caption_layers = caption_cfg.get("layer_indices", [])
    if image_layers != caption_layers:
        raise ValueError(
            "language_image and language_caption layer indices are not aligned in qwen_sweep_config.json."
        )

    selected_benchmarks = args.benchmarks if args.benchmarks is not None else image_benchmarks
    selected_indices = args.layer_indices if args.layer_indices is not None else image_layers
    selected_layers = [f"model.language_model.layers.{idx}" for idx in selected_indices]

    output_root = ensure_output_root(args.output_root)
    env = os.environ.copy()
    env["RESULTS_PATH"] = str(output_root)

    base_command = [
        sys.executable,
        "run.py",
        "--metric",
        args.metric,
        "--batch-size",
        str(args.batch_size),
        "--joint-models",
        "qwen3_vl_8b_img",
        "qwen3_vl_8b_txt",
        "--joint-stimuli",
        "image",
        "caption",
    ]

    commands: list[list[str]] = []
    for benchmark in selected_benchmarks:
        for layer_name in selected_layers:
            commands.append(
                base_command
                + [
                    "--benchmark",
                    benchmark,
                    "--joint-layers",
                    layer_name,
                    layer_name,
                ]
            )

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "sweep_type": "qwen_joint_language",
        "config_path": str(Path(args.config).expanduser().resolve()),
        "benchmarks": selected_benchmarks,
        "metric": args.metric,
        "batch_size": args.batch_size,
        "selected_layer_indices": selected_indices,
        "selected_layers": selected_layers,
        "joint_models": ["qwen3_vl_8b_img", "qwen3_vl_8b_txt"],
        "joint_stimuli": ["image", "caption"],
        "output_root": str(output_root),
        "commands": commands,
    }
    write_manifest(output_root / "qwen_joint_language_sweep_manifest.json", manifest)

    if args.dry_run:
        print("Dry run commands:")
        for cmd in commands:
            print(" ".join(cmd))
        return 0

    run_log_path = output_root / "qwen_joint_language_sweep_run_log.json"
    run_log: list[dict] = []
    failed_commands = []

    print(f"Writing raw results under: {output_root / 'results'}")
    for cmd in commands:
        print("Running command:")
        print(" ".join(cmd))
        started_utc = datetime.now(UTC).isoformat()
        benchmark_name = cmd[cmd.index("--benchmark") + 1]
        layer_name = cmd[cmd.index("--joint-layers") + 1]
        try:
            subprocess.run(cmd, check=True, env=env)
            status = "ok"
            return_code = 0
        except subprocess.CalledProcessError as exc:
            status = "failed"
            return_code = exc.returncode
            failed_commands.append({"benchmark": benchmark_name, "layer": layer_name})
            print(
                f"Command failed with return code {return_code}. "
                "See the Slurm log above for the exact failing command output."
            )
            if not args.continue_on_error:
                run_log.append(
                    {
                        "benchmark": benchmark_name,
                        "layer": layer_name,
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
                "benchmark": benchmark_name,
                "layer": layer_name,
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
        benchmarks=selected_benchmarks,
        selected_layers=selected_layers,
        metric=args.metric,
    )
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")
    if failed_commands:
        print(f"Failed runs: {failed_commands}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

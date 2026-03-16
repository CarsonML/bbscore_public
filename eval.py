#!/usr/bin/env python3
import os
import subprocess
import pickle
import argparse

# ——— parse outer-script CLI ———
parser = argparse.ArgumentParser(
    description="Wrapper to run run.py over many benchmarks/layers/models."
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Pass --debug through to run.py to enable debug mode there."
)
parser.add_argument(
    "--random-projection",
    type=str,
    default=None
)
parser.add_argument(
    "--random-projection-target-dim",
    type=int,
    default=None
)
args = parser.parse_args()

# Check if SCIKIT_LEARN_DATA environment variable is set.
if "SCIKIT_LEARN_DATA" not in os.environ:
    datahome = input(
        "SCIKIT_LEARN_DATA environment variable is not set. Please enter datahome directory: "
    )
    os.environ["SCIKIT_LEARN_DATA"] = datahome

# Define your experiment configurations
# Each entry has a list of (model, layer) tuples that share the same benchmarks and metrics
experiments = [
    {
        "model_layer_pairs": [
            ("convnext_tiny_imagenet_full_seed-0", "classifier.1"),
            ("convnext_large_imagenet_full_seed-0", "classifier.1"),
            ("convnext_base_imagenet_full_seed-0", "classifier.1"),
            ("convnext_small_imagenet_100_seed-0", "classifier.1"),
            ("convnext_small_imagenet_10_seed-0", "classifier.1"),
            ("convnext_small_imagenet_1_seed-0", "classifier.1"),
            ("alexnet_sin", "_orig_mod.classifier.4"),
            ("alexnet_trained", "_orig_mod.classifier.4"),
        ],
        "benchmarks": [
            "OnlinePhysionIntraContactPrediction",
            "OnlinePhysionContactPrediction"
        ],
        "metrics": ["physion_contact_prediction"]
    },
    {
        "model_layer_pairs": [
            ("convnext_tiny_imagenet_full_seed-0", "classifier.1"),
            ("convnext_large_imagenet_full_seed-0", "classifier.1"),
            ("convnext_base_imagenet_full_seed-0", "classifier.1"),
            ("convnext_small_imagenet_100_seed-0", "classifier.1"),
            ("convnext_small_imagenet_10_seed-0", "classifier.1"),
            ("convnext_small_imagenet_1_seed-0", "classifier.1"),
            ("convnext_small_imagenet_full_seed-0", "classifier.1"),
            ("alexnet_sin", "_orig_mod.classifier.4"),
            ("alexnet_trained", "_orig_mod.classifier.4"),
        ],
        "benchmarks": [
            "OnlinePhysionIntraContactDetection",
            "OnlinePhysionContactDetection",
        ],
        "metrics": ["physion_contact_detection"]
    },
    # Add more experiment configurations as needed
    # {
    #     "model_layer_pairs": [
    #         ("model_a", "layer1"),
    #         ("model_b", "layer3"),
    #     ],
    #     "benchmarks": ["Benchmark1", "Benchmark2"],
    #     "metrics": ["metric1", "metric2"]
    # },
]

# Define synonym groups so that rsa ⇔ stream_rsa and ridge ⇔ chunked_ridge are treated the same
metric_synonyms = {
    "chunked_ridge": {"chunked_ridge", "ridge"},
}

# Run each experiment configuration
for exp in experiments:
    model_layer_pairs = exp["model_layer_pairs"]
    benchmarks = exp["benchmarks"]
    all_metrics = exp["metrics"]

    for model, layer in model_layer_pairs:
        for benchmark in benchmarks:
            output_dir = os.path.join(os.environ["RESULTS_PATH"], "results")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"{model}_{layer}_{benchmark}.pkl"
            )

            # start with the full set of metrics we might run
            metrics_to_run = list(all_metrics)

            if os.path.exists(output_file):
                try:
                    with open(output_file, "rb") as f:
                        prev = pickle.load(f)

                    raw = prev.get("metrics", {})
                    # normalize into a list of dicts
                    if isinstance(raw, dict):
                        metrics_dicts = [raw]
                    elif isinstance(raw, list):
                        metrics_dicts = raw
                    else:
                        metrics_dicts = []

                    # collect all metric‐names seen so far
                    done = set()
                    for md in metrics_dicts:
                        if isinstance(md, dict):
                            done.update(md.keys())

                    # figure out which are still missing, considering synonyms
                    missing = []
                    for m in all_metrics:
                        syn_set = metric_synonyms.get(m, {m})
                        # if none of the synonyms have been done, schedule it
                        if not done.intersection(syn_set):
                            missing.append(m)

                    if not missing:
                        print(
                            f"Skipping {output_file}: all metrics already present "
                            f"({', '.join(sorted(done))})"
                        )
                        continue

                    print(
                        f"{output_file} has {', '.join(sorted(done))}; "
                        f"running only missing: {', '.join(missing)}"
                    )
                    metrics_to_run = missing

                except Exception as e:
                    print(
                        f"Warning: could not load existing results, overwriting: {e}"
                    )
                    # leave metrics_to_run as the full list

            # propagate debug flag if outer script was run with --debug
            if args.debug:
                print('⚠️ DEBUG MODE ACTIVATED! No doc will be saved to MongoDB!!')
            debug_arg = ["--debug"] if args.debug else []

            if args.random_projection:
                debug_arg += ["--random-projection", args.random_projection]
            if args.random_projection_target_dim is not None:
                debug_arg += [
                    "--random-projection-target-dim",
                    str(args.random_projection_target_dim)
                ]

            # build and run
            cmd = [
                "python", "run.py",
                "--model",     model,
                "--layer",     layer,
                "--benchmark", benchmark,
                "--metric",    *metrics_to_run,
                "--batch-size", str(32),
                *debug_arg,
            ]

            print("Running:", " ".join(cmd))
            r = subprocess.run(cmd)
            if r.returncode:
                print("FAIL:", " ".join(cmd))

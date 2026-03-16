import argparse
import os
import inspect
from typing import List, Optional, Union

from benchmarks import BENCHMARK_REGISTRY
from benchmarks.BBS import JointBenchmarkScore
from metrics import METRICS, validate_metric_benchmark, get_compatible_metrics
from models import MODEL_REGISTRY
from data.NSDShared import (
    NSDStimulusSet,
    NSDAssemblyV1,
    NSDAssemblyV1d,
    NSDAssemblyV1v,
    NSDAssemblyV2,
    NSDAssemblyV2d,
    NSDAssemblyV2v,
    NSDAssemblyV3,
    NSDAssemblyV3d,
    NSDAssemblyV3v,
    NSDAssemblyV4,
    NSDAssemblyLateral,
    NSDAssemblyVentral,
    NSDAssemblyParietal,
    NSDAssemblyMidLateral,
    NSDAssemblyMidVentral,
    NSDAssemblyMidParietal,
    NSDAssemblyHighLateral,
    NSDAssemblyHighVentral,
    NSDAssemblyHighParietal,
)
from data.NSDCaptions import NSDCaptionStimulusSet


NSD_JOINT_ASSEMBLY_MAP = {
    "NSDV1Shared": NSDAssemblyV1,
    "NSDV1dShared": NSDAssemblyV1d,
    "NSDV1vShared": NSDAssemblyV1v,
    "NSDV2Shared": NSDAssemblyV2,
    "NSDV2dShared": NSDAssemblyV2d,
    "NSDV2vShared": NSDAssemblyV2v,
    "NSDV3Shared": NSDAssemblyV3,
    "NSDV3dShared": NSDAssemblyV3d,
    "NSDV3vShared": NSDAssemblyV3v,
    "NSDV4Shared": NSDAssemblyV4,
    "NSDLateralShared": NSDAssemblyLateral,
    "NSDVentralShared": NSDAssemblyVentral,
    "NSDParietalShared": NSDAssemblyParietal,
    "NSDMidLateralShared": NSDAssemblyMidLateral,
    "NSDMidVentralShared": NSDAssemblyMidVentral,
    "NSDMidParietalShared": NSDAssemblyMidParietal,
    "NSDHighLateralShared": NSDAssemblyHighLateral,
    "NSDHighVentralShared": NSDAssemblyHighVentral,
    "NSDHighParietalShared": NSDAssemblyHighParietal,
    "NSDV1CaptionShared": NSDAssemblyV1,
    "NSDV1dCaptionShared": NSDAssemblyV1d,
    "NSDV1vCaptionShared": NSDAssemblyV1v,
    "NSDV2CaptionShared": NSDAssemblyV2,
    "NSDV2dCaptionShared": NSDAssemblyV2d,
    "NSDV2vCaptionShared": NSDAssemblyV2v,
    "NSDV3CaptionShared": NSDAssemblyV3,
    "NSDV3dCaptionShared": NSDAssemblyV3d,
    "NSDV3vCaptionShared": NSDAssemblyV3v,
    "NSDV4CaptionShared": NSDAssemblyV4,
    "NSDLateralCaptionShared": NSDAssemblyLateral,
    "NSDVentralCaptionShared": NSDAssemblyVentral,
    "NSDParietalCaptionShared": NSDAssemblyParietal,
    "NSDMidLateralCaptionShared": NSDAssemblyMidLateral,
    "NSDMidVentralCaptionShared": NSDAssemblyMidVentral,
    "NSDMidParietalCaptionShared": NSDAssemblyMidParietal,
    "NSDHighLateralCaptionShared": NSDAssemblyHighLateral,
    "NSDHighVentralCaptionShared": NSDAssemblyHighVentral,
    "NSDHighParietalCaptionShared": NSDAssemblyHighParietal,
}

JOINT_STIMULUS_CLASS_MAP = {
    "image": NSDStimulusSet,
    "caption": NSDCaptionStimulusSet,
}


def test_pipeline(
    model_identifier: str,
    layer_name: Union[str, List[str]],
    benchmark_identifier: str,
    metric_names: List[str],
    batch_size: Union[int, List[int]],
    debug: bool,
    use_ridge_smart_memory: bool,
    random_projection: str,
    aggregation_mode: str,
    joint_model_identifiers: Optional[List[str]] = None,
    joint_layer_names: Optional[List[str]] = None,
    joint_stimulus_modes: Optional[List[str]] = None,
):
    """
    Tests the benchmark pipeline with a given model, layer(s), and benchmark.
    """
    if joint_model_identifiers:
        print(
            f"Testing joint run with models: {joint_model_identifiers}, "
            f"layers: {joint_layer_names}, stimuli: {joint_stimulus_modes}, "
            f"benchmark: {benchmark_identifier}, aggregation: {aggregation_mode}"
        )
    else:
        layer_str = layer_name if isinstance(
            layer_name, str) else ", ".join(layer_name)
        print(
            f"Testing with model: {model_identifier}, layer(s): [{layer_str}], "
            f"benchmark: {benchmark_identifier}, aggregation: {aggregation_mode}"
        )

    # 1. Extract Benchmark Class from the Registry
    if benchmark_identifier not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Benchmark identifier '{benchmark_identifier}' not found in registry."
        )
    benchmark_class = BENCHMARK_REGISTRY[benchmark_identifier]

    # 2. Detect Online vs Offline Benchmark
    # We check the Method Resolution Order (MRO) to see if 'OnlineBenchmarkScore' is a parent.
    # This is more robust than signature inspection as subclasses might use **kwargs.
    is_online = any(
        c.__name__ == 'OnlineBenchmarkScore' for c in benchmark_class.__mro__)

    # Fallback: Check signature if MRO check fails (unlikely, but safe)
    if not is_online:
        try:
            init_sig = inspect.signature(benchmark_class)
            if 'dataloader_batch_size' in init_sig.parameters:
                is_online = True
        except ValueError:
            pass

    # Handle Layer Name
    if isinstance(layer_name, list) and len(layer_name) == 1:
        layer_arg = layer_name[0]
    else:
        layer_arg = layer_name

    # Warning for Online benchmarks if multiple layers are passed
    if is_online and isinstance(layer_arg, list):
        print(f"Warning: OnlineBenchmarkScore detected but multiple layers provided: {layer_arg}. "
              "This usually causes errors as OnlineFeatureExtractor expects a single layer string.")

    print(
        f"Instantiating {benchmark_identifier} (Type: {'Online' if is_online else 'Offline'})...")

    if joint_model_identifiers:
        if benchmark_identifier not in NSD_JOINT_ASSEMBLY_MAP:
            raise ValueError(
                "Joint run.py mode currently supports NSD shared and caption benchmarks only. "
                f"Unsupported benchmark: {benchmark_identifier}"
            )
        if joint_layer_names is None or len(joint_model_identifiers) != len(joint_layer_names):
            raise ValueError(
                "joint_layer_names must be provided and aligned one-to-one with joint_model_identifiers."
            )
        if joint_stimulus_modes is None:
            default_mode = "caption" if "CaptionShared" in benchmark_identifier else "image"
            joint_stimulus_modes = [default_mode] * len(joint_model_identifiers)
        if len(joint_stimulus_modes) != len(joint_model_identifiers):
            raise ValueError(
                "joint_stimulus_modes must be aligned one-to-one with joint_model_identifiers."
            )
        stimulus_classes = []
        for mode in joint_stimulus_modes:
            if mode not in JOINT_STIMULUS_CLASS_MAP:
                raise ValueError(
                    f"Unsupported joint stimulus mode '{mode}'. "
                    f"Supported modes: {sorted(JOINT_STIMULUS_CLASS_MAP.keys())}"
                )
            stimulus_classes.append(JOINT_STIMULUS_CLASS_MAP[mode])

        pipeline = JointBenchmarkScore(
            stimulus_train_class=stimulus_classes,
            model_identifiers=joint_model_identifiers,
            layer_names_per_model=joint_layer_names,
            assembly_class=NSD_JOINT_ASSEMBLY_MAP[benchmark_identifier],
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
            random_projection=random_projection,
        )
        pipeline.benchmark_identifier = benchmark_identifier
    elif model_identifier != 'None':
        pipeline = benchmark_class(
            model_identifier,
            layer_arg,
            batch_size=batch_size,
            debug=debug
        )
    else:
        pipeline = benchmark_class(debug=debug)
    # 5. Configure Pipeline Options (Conditionals for API differences)

    # Set Aggregation Mode
    if hasattr(pipeline, 'initialize_aggregation'):
        pipeline.initialize_aggregation(aggregation_mode)
        if aggregation_mode != "none":
            print(f"Aggregation mode set to: {aggregation_mode}")
    elif aggregation_mode != "none":
        print(
            f"Warning: Benchmark '{benchmark_identifier}' does not support aggregation. Ignoring mode '{aggregation_mode}'.")

    # Set Memory Optimization
    if hasattr(pipeline, 'use_ridge_smart_memory'):
        pipeline.use_ridge_smart_memory = use_ridge_smart_memory
        if use_ridge_smart_memory:
            print("Ridge smart memory feature is ENABLED.")

    # Set Random Projection
    if hasattr(pipeline, 'initialize_rp'):
        if random_projection:
            pipeline.initialize_rp(random_projection)
            print(f"Using {random_projection} Random Projection.")
    elif random_projection:
        print(
            f"Warning: Benchmark '{benchmark_identifier}' does not support Random Projection. Ignoring.")

    # 6. Add Desired Metrics (with compatibility check)
    for metric_name in metric_names:
        if not validate_metric_benchmark(metric_name, benchmark_identifier):
            compatible = get_compatible_metrics(benchmark_identifier)
            print(
                f"Warning: Metric '{metric_name}' may not be compatible "
                f"with benchmark '{benchmark_identifier}'.\n"
                f"  Compatible metrics: {compatible}"
            )
        pipeline.add_metric(metric_name)

    # 7. Run the Pipeline
    results = pipeline.run()

    print("\n--- Results ---")
    print("Benchmark Results:")

    # Results formatting
    if isinstance(results, dict):
        if 'metrics' in results:
            # Standard single result block (online or offline aggregated)
            # Online usually returns {'metrics': {...}, 'ceiling': ..., ...}
            for key, value in results.items():
                print(f"  {key}: {value}")
        else:
            # Multi-layer offline result (dict of dicts)
            for layer, res in results.items():
                print(f">>> Layer: {layer}")
                if isinstance(res, dict):
                    for key, value in res.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {res}")
    else:
        print(results)

    print("Pipeline test completed.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description=(
            "Run the BBScore benchmark pipeline with a specified model, layer, "
            "benchmark, and metric."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="videomae_base",
        help=f"Model identifier. Available models: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--layer",
        type=str,
        nargs="+",
        required=False,
        help="Layer name(s) to extract features from (e.g., 'encoder.layer.11' 'encoder.layer.10')."
    )
    parser.add_argument(
        "--joint-models",
        type=str,
        nargs="+",
        default=None,
        help="Model identifiers for a joint run. Use with --joint-layers and optional --joint-stimuli.",
    )
    parser.add_argument(
        "--joint-layers",
        type=str,
        nargs="+",
        default=None,
        help="One layer name per joint model for a joint run.",
    )
    parser.add_argument(
        "--joint-stimuli",
        type=str,
        nargs="+",
        default=None,
        help="Stimulus modality per joint model (currently: image or caption).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="NSDV1Shared",
        help=f"Benchmark identifier. Available benchmarks: {', '.join(BENCHMARK_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["ridge"],
        help=f"One or more metric identifiers. Available metrics: {', '.join(METRICS.keys())}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[4],
        help="Batch size: N or N N' (e.g. --batch-size 4 10).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for more detailed logging."
    )
    parser.add_argument(
        "--use-ridge-smart-memory",
        action="store_true",
        help="Enable smart memory estimation for ridge regression.",
    )
    parser.add_argument(
        "--random-projection",
        type=str,
        default=None,
        help="Enable random projection (dense, sparse).",
    )
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        default="none",
        choices=["none", "concatenate", "stack"],
        help="How to combine layers: 'none' (separate), 'concatenate' (feature concat), 'stack' (new dim).",
    )

    args = parser.parse_args()

    if args.joint_models:
        if args.joint_layers is None:
            parser.error("--joint-layers is required when --joint-models is used.")
        if len(args.joint_models) != len(args.joint_layers):
            parser.error("--joint-models and --joint-layers must have the same length.")
        if args.joint_stimuli is not None and len(args.joint_stimuli) != len(args.joint_models):
            parser.error("--joint-stimuli must have the same length as --joint-models.")
    elif args.layer is None:
        parser.error("--layer is required unless --joint-models is used.")

    # Normalize layer argument if single string passed despite nargs
    layers = args.layer
    if isinstance(layers, list) and len(layers) == 1:
        if " " in layers[0]:
            layers = layers[0].split()

    # --- Execute the Pipeline ---
    test_pipeline(
        model_identifier=args.model,
        layer_name=layers,
        benchmark_identifier=args.benchmark,
        metric_names=args.metric,
        batch_size=args.batch_size,
        debug=args.debug,
        use_ridge_smart_memory=args.use_ridge_smart_memory,
        random_projection=args.random_projection,
        aggregation_mode=args.aggregation_mode,
        joint_model_identifiers=args.joint_models,
        joint_layer_names=args.joint_layers,
        joint_stimulus_modes=args.joint_stimuli,
    )

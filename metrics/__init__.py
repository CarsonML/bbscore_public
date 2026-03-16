from .base import BaseMetric
from .base_online import OnlineMetric
from .ridge import RidgeMetric, TorchRidgeMetric, Ridge3DChunkedMetric, InverseRidgeChunkedMetric
from .ridge import RidgeAutoMetric, TorchLassoMetric, TorchElasticMetric, JointRidgeMetric
from .pls import PLSMetric
from .bidirectional import BidirectionalMappingMetric
from .one_to_one import OneToOneMappingMetric
from .soft_matching import SoftMatchingMetric
from .semi_matching import SemiMatchingMetric
from .rsa import RSAMetric, TemporalRSAMetric
from .versa import VeRSAMetric
from .online_mappers import OnlineLinearClassifier, OnlineLinearRegressor
from .online_mappers import OnlineTransformerClassifier
from .orientation_selectivity import OrientationSelectivity
from .physion import OnlinePhysionContactDetection, OnlinePhysionContactPrediction, OnlinePhysionPlacementDetection, OnlinePhysionPlacementPrediction


METRICS = {
    "ridge": RidgeMetric,
    "joint_ridge": JointRidgeMetric,
    "torch_ridge": TorchRidgeMetric,
    "torch_lasso": TorchLassoMetric,
    "torch_elastic": TorchElasticMetric,
    "pls": PLSMetric,
    "bidirectional": BidirectionalMappingMetric,
    "one_to_one": OneToOneMappingMetric,
    "soft_matching": SoftMatchingMetric,
    "semi_matching": SemiMatchingMetric,
    "rsa": RSAMetric,
    "temporal_rsa": TemporalRSAMetric,
    "versa": VeRSAMetric,
    "temporal_ridge": Ridge3DChunkedMetric,
    "inverse_ridge": InverseRidgeChunkedMetric,

    # online metrics
    "online_linear_classifier": OnlineLinearClassifier,
    "online_linear_regressor": OnlineLinearRegressor,
    "online_transformer_classifier": OnlineTransformerClassifier,
    "physion_placement_prediction": OnlinePhysionPlacementPrediction,
    "physion_placement_detection": OnlinePhysionPlacementDetection,
    "physion_contact_prediction": OnlinePhysionContactPrediction,
    "physion_contact_detection": OnlinePhysionContactDetection,
    # topographic metrics
    "orientation_selectivity": OrientationSelectivity,
}


# ---------------------------------------------------------------------------
# Metric-Benchmark Compatibility
# ---------------------------------------------------------------------------
# Groups of metrics by the type of analysis they perform.
METRIC_GROUPS = {
    "offline_regression": [
        "ridge", "joint_ridge", "torch_ridge", "torch_lasso", "torch_elastic",
        "pls", "temporal_ridge", "inverse_ridge",
    ],
    "offline_similarity": [
        "rsa", "temporal_rsa", "versa",
        "bidirectional", "one_to_one", "soft_matching", "semi_matching",
    ],
    "online_regression": [
        "online_linear_regressor",
    ],
    "online_classification": [
        "online_linear_classifier", "online_transformer_classifier",
    ],
    "physion_contact": [
        "physion_contact_prediction", "physion_contact_detection",
    ],
    "physion_placement": [
        "physion_placement_prediction", "physion_placement_detection",
    ],
    "topographic": [
        "orientation_selectivity",
    ],
    # TR-level benchmarks are standalone classes with their own GroupKFold
    # ridge regression (sklearn RidgeCV). They do NOT use the registry's
    # "temporal_ridge" (Ridge3DChunkedMetric), which expects 3D chunked
    # features from the standard BenchmarkScore pipeline.
    "tr_level": [
        "ridge", "temporal_rsa",
    ],
}

# Maps benchmark name prefixes to their compatible metric groups.
# Checked in order; first matching prefix wins.
_BENCHMARK_METRIC_MAP = [
    # TR-level LeBel (must be before generic LeBel)
    ("LeBel2023AudioTR",  ["tr_level"]),
    ("LeBel2023TR",       ["tr_level"]),
    # Audio average LeBel (must be before generic LeBel)
    ("LeBel2023Audio",    ["offline_regression", "offline_similarity"]),
    # Offline neural benchmarks
    ("NSD",               ["offline_regression", "offline_similarity"]),
    ("TVSD",              ["offline_regression", "offline_similarity"]),
    ("BMD",               ["offline_regression", "offline_similarity"]),
    ("LeBel2023",         ["offline_regression", "offline_similarity"]),
    ("V1",                ["offline_regression", "offline_similarity",
                           "topographic"]),
    ("PhysionContact",    ["offline_regression", "offline_similarity"]),
    ("PhysionPlacement",  ["offline_regression", "offline_similarity"]),
    ("PhysionIntra",      ["offline_regression", "offline_similarity"]),
    # Online benchmarks
    ("OnlineTVSD",        ["online_regression"]),
    ("OnlinePhysionContact",    ["online_classification",
                                 "physion_contact"]),
    ("OnlinePhysionIntraContact", ["online_classification",
                                   "physion_contact"]),
    ("OnlinePhysionPlacement",  ["online_classification",
                                 "physion_placement"]),
    ("OnlinePhysionIntraPlacement", ["online_classification",
                                     "physion_placement"]),
    ("SSV2",              ["online_classification"]),
    ("AugmentedSSV2",     ["online_classification"]),
]


def get_compatible_metrics(benchmark_name):
    """Return list of compatible metric names for a given benchmark.

    Args:
        benchmark_name: Name of the benchmark (e.g. 'NSDV1Shared').

    Returns:
        List of metric name strings, or None if benchmark is unknown.
    """
    for prefix, groups in _BENCHMARK_METRIC_MAP:
        if benchmark_name.startswith(prefix):
            compatible = []
            for g in groups:
                compatible.extend(METRIC_GROUPS.get(g, []))
            return list(dict.fromkeys(compatible))  # dedupe, keep order
    return None


def validate_metric_benchmark(metric_name, benchmark_name):
    """Check if a metric is compatible with a benchmark.

    Returns:
        True if compatible or unknown benchmark, False if incompatible.
    """
    compatible = get_compatible_metrics(benchmark_name)
    if compatible is None:
        return True  # unknown benchmark — don't block
    return metric_name in compatible


__all__ = ["BaseMetric", "OnlineMetric", "METRICS",
           "get_compatible_metrics", "validate_metric_benchmark"]

"""
NSD shared benchmarks with image + caption in a single multimodal forward (Qwen3-VL joint).
Uses NSDImageCaptionStimulusSet; set NSD_CAPTIONS_PATH to the desired JSONL.

Batch size > 1 is supported: each sample is preprocessed once, then tensor dicts are merged
in the feature extractor (same pattern as image/caption-only Qwen runs).
"""

from benchmarks.BBS import BenchmarkScore
from benchmarks import BENCHMARK_REGISTRY
from data.NSDCaptions import NSDImageCaptionStimulusSet
from data.NSDShared import (
    NSDAssemblyV1,
    NSDAssemblyV2,
    NSDAssemblyV3,
    NSDAssemblyV4,
    NSDAssemblyLateral,
    NSDAssemblyParietal,
    NSDAssemblyHighLateral,
    NSDAssemblyHighVentral,
)


class NSDV1SharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV1SharedMultimodal"] = NSDV1SharedMultimodal


class NSDV2SharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV2SharedMultimodal"] = NSDV2SharedMultimodal


class NSDV3SharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV3SharedMultimodal"] = NSDV3SharedMultimodal


class NSDV4SharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV4,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV4SharedMultimodal"] = NSDV4SharedMultimodal


class NSDParietalSharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDParietalSharedMultimodal"] = NSDParietalSharedMultimodal


class NSDLateralSharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDLateralSharedMultimodal"] = NSDLateralSharedMultimodal


class NSDHighVentralSharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDHighVentralSharedMultimodal"] = NSDHighVentralSharedMultimodal


class NSDHighLateralSharedMultimodal(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDImageCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDHighLateralSharedMultimodal"] = NSDHighLateralSharedMultimodal

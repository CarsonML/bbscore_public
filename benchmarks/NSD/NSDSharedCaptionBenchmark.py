from benchmarks.BBS import BenchmarkScore
from benchmarks import BENCHMARK_REGISTRY
from data.NSDCaptions import NSDCaptionStimulusSet
from data.NSDShared import (
    NSDAssemblyV1,
    NSDAssemblyV1d,
    NSDAssemblyV1v,
    NSDAssemblyV2,
    NSDAssemblyV2v,
    NSDAssemblyV2d,
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


class NSDV1CaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV1CaptionShared"] = NSDV1CaptionShared


class NSDV1dCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1d,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV1dCaptionShared"] = NSDV1dCaptionShared


class NSDV1vCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV1v,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV1vCaptionShared"] = NSDV1vCaptionShared


class NSDV2CaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV2CaptionShared"] = NSDV2CaptionShared


class NSDV2dCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2d,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV2dCaptionShared"] = NSDV2dCaptionShared


class NSDV2vCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV2v,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV2vCaptionShared"] = NSDV2vCaptionShared


class NSDV3CaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV3CaptionShared"] = NSDV3CaptionShared


class NSDV3dCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3d,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV3dCaptionShared"] = NSDV3dCaptionShared


class NSDV3vCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV3v,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV3vCaptionShared"] = NSDV3vCaptionShared


class NSDV4CaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyV4,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDV4CaptionShared"] = NSDV4CaptionShared


class NSDLateralCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDLateralCaptionShared"] = NSDLateralCaptionShared


class NSDVentralCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDVentralCaptionShared"] = NSDVentralCaptionShared


class NSDParietalCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDParietalCaptionShared"] = NSDParietalCaptionShared


class NSDHighLateralCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDHighLateralCaptionShared"] = NSDHighLateralCaptionShared


class NSDHighVentralCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDHighVentralCaptionShared"] = NSDHighVentralCaptionShared


class NSDHighParietalCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyHighParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDHighParietalCaptionShared"] = NSDHighParietalCaptionShared


class NSDMidLateralCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyMidLateral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDMidLateralCaptionShared"] = NSDMidLateralCaptionShared


class NSDMidVentralCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyMidVentral,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDMidVentralCaptionShared"] = NSDMidVentralCaptionShared


class NSDMidParietalCaptionShared(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=NSDCaptionStimulusSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            assembly_class=NSDAssemblyMidParietal,
            batch_size=batch_size,
            num_workers=0,
            debug=debug,
        )


BENCHMARK_REGISTRY["NSDMidParietalCaptionShared"] = NSDMidParietalCaptionShared


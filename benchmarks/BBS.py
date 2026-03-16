import os
import pickle
import configparser
import datetime
import getpass
import numpy as np
import pickle
import subprocess
import torch
import torch.nn as nn
import psutil
from torch.utils.data import DataLoader
from typing import Union, List

from sklearn.datasets import get_data_home

from extractor_wrapper import FeatureExtractor
from metrics import METRICS
from models import get_model_class_and_id, MODEL_REGISTRY
from data.utils import custom_collate  # custom collate function


def _run_git(cmd):
    try:
        out = subprocess.check_output(["git"] + cmd,
                                      stderr=subprocess.DEVNULL)
        return out.strip().decode()
    except Exception:
        return None


def get_local_commit():
    return _run_git(["rev-parse", "HEAD"])


def is_worktree_clean():
    status = _run_git(["status", "--porcelain"])
    return status == ""


def estimate_gram_bytes(N):
    return N * N * np.dtype(np.float64).itemsize


def get_mem_info():
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 2**30, vm.available / 2**30, vm.total / 2**30


class BenchmarkScore:
    def __init__(
        self,
        stimulus_train_class,
        model_identifier,
        layer_name: Union[str, List[str]] = None,
        stimulus_test_class=None,
        assembly_class=None,
        assembly_train_kwargs=None,
        assembly_test_kwargs=None,
        batch_size=32,
        num_workers=4,
        task='neural',
        save_features=False,
        debug=False,
        safety_factor=0.8,
        random_projection=None,
        # aggregation_mode removed from init to support inheritance cleanly
    ):

        self.debug = debug
        self.safety_factor = safety_factor
        self.random_projection = random_projection
        self.use_ridge_smart_memory = False
        self.aggregation_mode = "none"  # Default

        # Instantiate model and preprocessing
        self.model_class, self.model_id_mapping = get_model_class_and_id(
            model_identifier)
        self.model_instance = self.model_class()
        self.model_identifier = model_identifier

        # layer_name can be list or str
        self.layer_names = layer_name if isinstance(
            layer_name, list) else [layer_name]
        # Keep original for filename if single
        self.layer_name = layer_name

        # Prepare stimuli
        self.stimulus_train = stimulus_train_class(
            preprocess=self.model_instance.preprocess_fn
        )
        self.stimulus_test = None
        if stimulus_test_class is not None:
            self.stimulus_test = stimulus_test_class(
                preprocess=self.model_instance.preprocess_fn
            )

        if isinstance(batch_size, (list, tuple)):
            if len(batch_size) == 2:
                self.batch_size, self.test_batch_size = batch_size[0], batch_size[1]
            else:
                self.batch_size = batch_size[0]
                self.test_batch_size = None
        else:
            self.batch_size = int(batch_size)
            self.test_batch_size = None

        # Retrieve the model and create a feature extractor
        self.model = self.model_instance.get_model(self.model_id_mapping)
        self.extractor = FeatureExtractor(self.model, self.layer_names,
                                          postprocess_fn=self.model_instance.postprocess_fn,
                                          batch_size=self.batch_size,
                                          num_workers=num_workers,
                                          static=self.model_instance.static,
                                          aggregation_mode="none")  # Default init

        self.task = task
        self.metrics = {}
        self.metric_params = {}

        self.assembly_class = assembly_class
        self.assembly_train_kwargs = assembly_train_kwargs or {}
        self.assembly_test_kwargs = assembly_test_kwargs or {}

        data_home = get_data_home()
        self.features_path = os.path.join(data_home, 'features')
        os.makedirs(self.features_path, exist_ok=True)

        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.save_features = save_features

    def initialize_rp(self, rp):
        self.extractor.random_projection = rp

    def initialize_aggregation(self, mode):
        """
        Initialize aggregation mode after instantiation.
        This allows subclasses to inherit without modifying __init__ signatures.
        """
        self.aggregation_mode = mode
        self.extractor.aggregation_mode = mode

    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def _process_single_layer_result(self, features_train, features_test, labels_train, labels_test, current_layer_name):
        # Handle dict features (aggregation_mode="none") by extracting the current layer
        if isinstance(features_train, dict):
            if current_layer_name in features_train:
                features_train = features_train[current_layer_name]
            elif len(features_train) == 1:
                features_train = next(iter(features_train.values()))
            else:
                raise ValueError(
                    f"features_train is a dict with keys {list(features_train.keys())} "
                    f"but current_layer_name '{current_layer_name}' not found."
                )
        if isinstance(features_test, dict):
            if current_layer_name in features_test:
                features_test = features_test[current_layer_name]
            elif len(features_test) == 1:
                features_test = next(iter(features_test.values()))
            else:
                features_test = None
        stratify_labels_train = None
        if self.assembly_class:
            assembly = self.assembly_class()
            try:
                assembly_train_data = assembly.get_assembly(
                    **self.assembly_train_kwargs)
                if len(assembly_train_data) == 3:
                    target_train, ceiling, stratify_labels_train = assembly_train_data
                elif len(assembly_train_data) == 2:
                    target_train, ceiling = assembly_train_data
                else:
                    raise ValueError(
                        f"Assembly get_assembly returned {len(assembly_train_data)} values.")
            except Exception as e:
                print(f"Error calling get_assembly for training: {e}")
                raise

            if self.stimulus_test is not None:
                target_test, _ = assembly.get_assembly(
                    **self.assembly_test_kwargs)
            else:
                target_test = None
        else:
            target_train = labels_train
            target_test = labels_test
            ceiling = None
            stratify_labels_train = None

        feature_train_rows = (
            features_train.shape[0]
            if hasattr(features_train, "shape") and len(features_train.shape) > 0
            else None
        )
        target_train_rows = (
            target_train.shape[0]
            if hasattr(target_train, "shape") and len(target_train.shape) > 0
            else None
        )
        if (
            feature_train_rows is not None
            and target_train_rows is not None
            and feature_train_rows != target_train_rows
        ):
            raise ValueError(
                f"Sample count mismatch before metric computation for layer '{current_layer_name}': "
                f"features_train has shape {getattr(features_train, 'shape', None)}, "
                f"but target_train has shape {getattr(target_train, 'shape', None)}."
            )

        if features_test is not None and target_test is not None:
            feature_test_rows = (
                features_test.shape[0]
                if hasattr(features_test, "shape") and len(features_test.shape) > 0
                else None
            )
            target_test_rows = (
                target_test.shape[0]
                if hasattr(target_test, "shape") and len(target_test.shape) > 0
                else None
            )
            if (
                feature_test_rows is not None
                and target_test_rows is not None
                and feature_test_rows != target_test_rows
            ):
                raise ValueError(
                    f"Sample count mismatch before metric computation for layer '{current_layer_name}': "
                    f"features_test has shape {getattr(features_test, 'shape', None)}, "
                    f"but target_test has shape {getattr(target_test, 'shape', None)}."
                )

        results = {}
        n_metrics = len(self.metrics)
        for name, metric_class in self.metrics.items():
            try:
                extra = self.metric_params.get(name, {})
                if ceiling is not None:
                    metric_instance = metric_class(
                        ceiling=ceiling, **extra)
                else:
                    metric_instance = metric_class(**extra)
                results[name] = metric_instance.compute(
                    features_train,
                    target_train,
                    test_source=features_test,
                    test_target=target_test,
                    stratify_on=stratify_labels_train
                )
            except Exception as e:
                if n_metrics == 1:
                    raise RuntimeError(f"Metric '{name}' failed: {e}") from e
                print(f"⚠️ Metric '{name}' failed: {e}")
                continue

        results['timestamp'] = datetime.datetime.utcnow().isoformat()
        results['aggregation_mode'] = self.aggregation_mode

        benchmark_name = self.__class__.__name__
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{current_layer_name}_{benchmark_name}.pkl"
        )

        if os.path.exists(results_file):
            try:
                with open(results_file, 'rb') as f:
                    prev = pickle.load(f)
                prev_metrics = prev.get("metrics", [])
                if isinstance(prev_metrics, dict):
                    prev_metrics = [prev_metrics]
                prev_metrics.append(results)
                merged = {"metrics": prev_metrics, "ceiling": ceiling}
                if self.aggregation_mode != "none":
                    merged["constituent_layers"] = self.layer_names
            except Exception:
                merged = {"metrics": results, "ceiling": ceiling}
        else:
            merged = {"metrics": results, "ceiling": ceiling}
            if self.aggregation_mode != "none":
                merged["constituent_layers"] = self.layer_names

        with open(results_file, 'wb') as f:
            pickle.dump(merged, f)

        if self.save_features:
            stim_name = self.stimulus_train.__class__.__name__
            feat_file = os.path.join(
                self.features_path,
                f"{self.model_identifier}_{current_layer_name}_{stim_name}_features.pkl"
            )
            try:
                os.makedirs(os.path.dirname(feat_file), exist_ok=True)
                with open(feat_file, 'wb') as f:
                    pickle.dump({'train': features_train, 'train_labels': labels_train,
                                 'test': features_test, 'test_labels': labels_test}, f)
            except Exception as e:
                print(f"Error saving features: {e}")

        return results, ceiling

    def run(self):

        # --- Memory Estimation (Warmup) ---
        ridge_metrics_present = any(
            'ridge' in name.lower() for name in self.metrics)
        downsample_factor = 1.0

        if ridge_metrics_present and self.use_ridge_smart_memory:
            # 1. Warmup Extract
            warmup_loader = DataLoader(self.stimulus_train, batch_size=1, shuffle=False,
                                       num_workers=self.extractor.num_workers, collate_fn=custom_collate)
            batch = next(iter(warmup_loader))
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.extractor.get_activations(inputs)

            # 2. Estimate "Elements per Sample" based on Aggregation Mode
            elements_per_sample = 0
            if self.aggregation_mode in ["concatenate", "stack"]:
                # Use logic similar to extract_features but just for shape
                # We can hack this by calling process_sequence_features on the captured warmup features
                # But careful: features is a dict
                shapes = []
                for l_name in self.layer_names:
                    raw = self.extractor.features[l_name]
                    processed = self.extractor._process_sequence_features(raw)
                    if isinstance(processed, torch.Tensor):
                        processed = processed.cpu().numpy()
                    shapes.append(processed.shape)  # (B, [T], D)

                # Assuming B=1 from warmup
                if self.aggregation_mode == "concatenate":
                    # Sum of last dims
                    total_dim = sum(s[-1] for s in shapes)
                    # Other dims (e.g. T) are shared
                    temporal_dim = np.prod(
                        shapes[0][1:-1]) if len(shapes[0]) > 2 else 1
                    elements_per_sample = temporal_dim * total_dim
                elif self.aggregation_mode == "stack":
                    # New dim L, must have shared D (via RP target_dim)
                    # If RP is not set, we assume they match or it will fail later
                    target_d = self.extractor.target_dim if self.extractor.target_dim else shapes[
                        0][-1]
                    temporal_dim = np.prod(
                        shapes[0][1:-1]) if len(shapes[0]) > 2 else 1
                    elements_per_sample = temporal_dim * \
                        len(self.layer_names) * target_d
            else:
                # Mode = "none". Bottleneck is the largest single layer.
                max_elements = 0
                for l_name in self.layer_names:
                    raw = self.extractor.features[l_name]
                    processed = self.extractor._process_sequence_features(raw)
                    size = np.prod(processed.shape[1:])  # Exclude batch
                    if size > max_elements:
                        max_elements = size
                elements_per_sample = max_elements

            # 3. Calculate Budget (Same logic as original)
            rss, avail_gb, total_gb = get_mem_info()
            budget_bytes = avail_gb * (2**30) * self.safety_factor
            byte_f64 = np.dtype(np.float64).itemsize
            N_train = len(self.stimulus_train)
            N_test = len(self.stimulus_test) if self.stimulus_test else 0

            # Rough overhead model
            overhead_factor = 2.0
            effective_budget = budget_bytes / overhead_factor

            # Simple assumption: targets are small compared to features
            available_for_features = effective_budget
            cost_per_sample = elements_per_sample * byte_f64

            total_needed = (N_train + N_test) * cost_per_sample

            if total_needed > available_for_features:
                downsample_factor = available_for_features / total_needed
                print(
                    f"--- Smart Memory: Downsampling factor set to {downsample_factor:.4f} ---")
            else:
                print("--- Smart Memory: Sufficient memory, no downsampling. ---")

        # --- Extraction ---
        print("Extracting features...")
        features_train_raw, labels_train = self.extractor.extract_features(
            self.stimulus_train, downsample_factor)

        features_test_raw, labels_test = None, None
        if self.stimulus_test is not None:
            features_test_raw, labels_test = self.extractor.extract_features(
                self.stimulus_test, downsample_factor, self.test_batch_size)

        all_results = {}

        # CASE 1: Aggregated (Concatenate or Stack)
        if self.aggregation_mode in ["concatenate", "stack"]:
            if len(self.layer_names) == 1:
                combined_name = self.layer_names[0]
            else:
                base = "_".join(self.layer_names)
                if len(base) > 80:
                    import hashlib
                    h = hashlib.md5(base.encode()).hexdigest()[:8]
                    combined_name = f"{self.aggregation_mode.capitalize()}_{len(self.layer_names)}Layers_{h}"
                else:
                    combined_name = f"{self.aggregation_mode.capitalize()}_{base}"

            print(
                f"Running metrics for Aggregated ({self.aggregation_mode}): {combined_name}")

            res, ceil = self._process_single_layer_result(
                features_train_raw, features_test_raw, labels_train, labels_test, combined_name
            )
            return {'metrics': res, 'ceiling': ceil}

        # CASE 2: Separate Layers (Dict)
        else:
            if not isinstance(features_train_raw, dict):
                # Fallback for single layer legacy
                features_train_raw = {self.layer_names[0]: features_train_raw}
                if features_test_raw is not None:
                    features_test_raw = {
                        self.layer_names[0]: features_test_raw}

            for layer_key, f_train in features_train_raw.items():
                print(f"Running metrics for Separate Layer: {layer_key}")
                f_test = features_test_raw[layer_key] if features_test_raw else None
                res, ceil = self._process_single_layer_result(
                    f_train, f_test, labels_train, labels_test, layer_key
                )
                all_results[layer_key] = {'metrics': res, 'ceiling': ceil}

            return all_results


class JointBenchmarkScore(BenchmarkScore):
    """
    Benchmark that builds a joint feature space from multiple models.

    It extracts features from each model for the same stimuli, selects
    the requested layer per model (currently assumes one layer per
    model), concatenates the resulting feature matrices along the
    feature dimension, and then calls the standard metric machinery
    once on this joint feature matrix.
    """

    def __init__(
        self,
        stimulus_train_class,
        model_identifiers: List[str],
        layer_names_per_model: List[Union[str, List[str]]],
        stimulus_test_class=None,
        assembly_class=None,
        assembly_train_kwargs=None,
        assembly_test_kwargs=None,
        batch_size=32,
        num_workers=4,
        task: str = "neural",
        save_features: bool = False,
        debug: bool = False,
        safety_factor: float = 0.8,
        random_projection=None,
    ):
        if not isinstance(model_identifiers, (list, tuple)) or len(model_identifiers) < 2:
            raise ValueError(
                "JointBenchmarkScore expects at least two model_identifiers."
            )
        if len(model_identifiers) != len(layer_names_per_model):
            raise ValueError(
                "layer_names_per_model must have the same length as model_identifiers."
            )

        if isinstance(stimulus_train_class, (list, tuple)):
            if len(stimulus_train_class) != len(model_identifiers):
                raise ValueError(
                    "stimulus_train_class must be a single class or a list aligned with model_identifiers."
                )
            stimulus_train_classes = list(stimulus_train_class)
        else:
            stimulus_train_classes = [stimulus_train_class] * len(model_identifiers)

        if stimulus_test_class is None:
            stimulus_test_classes = [None] * len(model_identifiers)
        elif isinstance(stimulus_test_class, (list, tuple)):
            if len(stimulus_test_class) != len(model_identifiers):
                raise ValueError(
                    "stimulus_test_class must be None, a single class, or a list aligned with model_identifiers."
                )
            stimulus_test_classes = list(stimulus_test_class)
        else:
            stimulus_test_classes = [stimulus_test_class] * len(model_identifiers)

        first_layers = layer_names_per_model[0]
        super().__init__(
            stimulus_train_class=stimulus_train_classes[0],
            model_identifier=model_identifiers[0],
            layer_name=first_layers,
            stimulus_test_class=stimulus_test_classes[0],
            assembly_class=assembly_class,
            assembly_train_kwargs=assembly_train_kwargs,
            assembly_test_kwargs=assembly_test_kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            task=task,
            save_features=save_features,
            debug=debug,
            safety_factor=safety_factor,
            random_projection=random_projection,
        )

        self.joint_model_identifiers: List[str] = list(model_identifiers)
        self.joint_layer_names: List[List[str]] = []
        for ln in layer_names_per_model:
            if isinstance(ln, list):
                if len(ln) != 1:
                    raise NotImplementedError(
                        "JointBenchmarkScore currently supports exactly one layer per model."
                    )
                self.joint_layer_names.append(ln)
            else:
                self.joint_layer_names.append([ln])

        # Override model_identifier used in filenames to reflect joint setup
        self.model_identifier = "Joint_" + "_".join(self.joint_model_identifiers)

        # Build feature extractors for all models.
        self.joint_model_instances = [self.model_instance]
        self.joint_models = [self.model]
        self.joint_extractors = [self.extractor]
        self.joint_stimulus_trains = [self.stimulus_train]
        self.joint_stimulus_tests = [self.stimulus_test]
        self.joint_extractors[0].layer_names = self.joint_layer_names[0]

        for midx in range(1, len(self.joint_model_identifiers)):
            mid = self.joint_model_identifiers[midx]
            layer_names = self.joint_layer_names[midx]

            model_class, model_id_mapping = get_model_class_and_id(mid)
            model_instance = model_class()
            model = model_instance.get_model(model_id_mapping)

            extractor = FeatureExtractor(
                model,
                layer_names,
                postprocess_fn=model_instance.postprocess_fn,
                batch_size=self.batch_size,
                num_workers=num_workers,
                static=model_instance.static,
                aggregation_mode="none",
            )

            self.joint_model_instances.append(model_instance)
            self.joint_models.append(model)
            self.joint_extractors.append(extractor)
            self.joint_stimulus_trains.append(
                stimulus_train_classes[midx](preprocess=model_instance.preprocess_fn)
            )
            if stimulus_test_classes[midx] is not None:
                self.joint_stimulus_tests.append(
                    stimulus_test_classes[midx](preprocess=model_instance.preprocess_fn)
                )
            else:
                self.joint_stimulus_tests.append(None)

        self.use_ridge_smart_memory = False

    def initialize_rp(self, rp):
        for extractor in self.joint_extractors:
            extractor.random_projection = rp

    def initialize_aggregation(self, mode):
        if mode != "none":
            raise NotImplementedError(
                "JointBenchmarkScore currently supports aggregation_mode='none' only."
            )
        self.aggregation_mode = mode
        for extractor in self.joint_extractors:
            extractor.aggregation_mode = mode

    def run(self):
        print("Extracting joint features from models:", self.joint_model_identifiers)

        def _flatten_for_joint(features):
            if features is None:
                return None
            if hasattr(features, "ndim") and features.ndim > 2:
                return features.reshape(features.shape[0], -1)
            return features

        features_train_per_model = []
        features_test_per_model = []
        labels_train_ref = None
        labels_test_ref = None

        for idx, extractor in enumerate(self.joint_extractors):
            features_train_raw, labels_train = extractor.extract_features(
                self.joint_stimulus_trains[idx], 1.0
            )
            features_test_raw, labels_test = None, None
            if self.joint_stimulus_tests[idx] is not None:
                features_test_raw, labels_test = extractor.extract_features(
                    self.joint_stimulus_tests[idx], 1.0, self.test_batch_size
                )

            if labels_train_ref is None:
                labels_train_ref = labels_train
            else:
                if labels_train_ref is not None and labels_train is not None:
                    if len(labels_train_ref) != len(labels_train):
                        raise ValueError(
                            "Label length mismatch across models in JointBenchmarkScore."
                        )

            if labels_test_ref is None:
                labels_test_ref = labels_test
            else:
                if (
                    labels_test_ref is not None
                    and labels_test is not None
                    and len(labels_test_ref) != len(labels_test)
                ):
                    raise ValueError(
                        "Test label length mismatch across models in JointBenchmarkScore."
                    )

            layer_key = self.joint_layer_names[idx][0]
            if isinstance(features_train_raw, dict):
                if layer_key in features_train_raw:
                    f_train = features_train_raw[layer_key]
                elif len(features_train_raw) == 1:
                    f_train = next(iter(features_train_raw.values()))
                else:
                    raise ValueError(
                        f"features_train_raw keys {list(features_train_raw.keys())} "
                        f"do not contain requested layer '{layer_key}'."
                    )
            else:
                f_train = features_train_raw

            if features_test_raw is not None:
                if isinstance(features_test_raw, dict):
                    if layer_key in features_test_raw:
                        f_test = features_test_raw[layer_key]
                    elif len(features_test_raw) == 1:
                        f_test = next(iter(features_test_raw.values()))
                    else:
                        f_test = None
                else:
                    f_test = features_test_raw
            else:
                f_test = None

            features_train_per_model.append(_flatten_for_joint(f_train))
            features_test_per_model.append(_flatten_for_joint(f_test))

        features_train_joint = None
        for f in features_train_per_model:
            if f is None:
                continue
            if features_train_joint is None:
                features_train_joint = f
            else:
                features_train_joint = np.concatenate(
                    [features_train_joint, f], axis=1
                )

        features_test_joint = None
        for f in features_test_per_model:
            if f is None:
                continue
            if features_test_joint is None:
                features_test_joint = f
            else:
                features_test_joint = np.concatenate(
                    [features_test_joint, f], axis=1
                )

        if features_train_joint is None:
            raise RuntimeError(
                "No joint training features were constructed in JointBenchmarkScore."
            )

        combined_name = "Joint_" + "_".join(
            f"{mid}:{self.joint_layer_names[idx][0]}"
            for idx, mid in enumerate(self.joint_model_identifiers)
        )

        print(f"Running metrics for Joint features: {combined_name}")

        res, ceil = self._process_single_layer_result(
            features_train_joint,
            features_test_joint,
            labels_train_ref,
            labels_test_ref,
            combined_name,
        )
        return {"metrics": res, "ceiling": ceil}


class AssemblyBenchmarkScorer:
    def __init__(
        self,
        source_assembly_class,
        target_assembly_class,
        source_assembly_train_kwargs=None,
        source_assembly_test_kwargs=None,
        target_assembly_train_kwargs=None,
        target_assembly_test_kwargs=None,
        task='neural',
        debug=False,
    ):
        self.debug = debug
        self.task = task
        self.metrics = {}
        self.metric_params = {}

        self.source_assembly_class = source_assembly_class
        self.target_assembly_class = target_assembly_class
        self.source_assembly_train_kwargs = source_assembly_train_kwargs or {}
        self.source_assembly_test_kwargs = source_assembly_test_kwargs or {}
        self.target_assembly_train_kwargs = target_assembly_train_kwargs or {}
        self.target_assembly_test_kwargs = target_assembly_test_kwargs or {}

        self.source_name = self.source_assembly_class.__name__
        self.target_name = self.target_assembly_class.__name__

        data_home = get_data_home()
        results_base = os.environ.get('RESULTS_PATH', data_home)
        results_dir = os.path.join(results_base, 'results')
        os.makedirs(results_dir, exist_ok=True)

        benchmark_class_name = self.__class__.__name__
        self.results_file = os.path.join(
            results_dir,
            f"{benchmark_class_name}.pkl"
        )

    def initialize_rp(self, rp):
        if rp is not None:
            print(
                "Warning: Random Projection is not supported for AssemblyBenchmarkScorer. Ignoring.")

    def initialize_aggregation(self, mode):
        if mode != "none":
            print(
                "Warning: Layer Aggregation is not supported for AssemblyBenchmarkScorer. Ignoring.")

    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def run(self):

        # 1. Load source
        source_assembly = self.source_assembly_class()
        source_train, _ = source_assembly.get_assembly(
            **self.source_assembly_train_kwargs)
        source_test, _ = (source_assembly.get_assembly(**self.source_assembly_test_kwargs)
                          if self.source_assembly_test_kwargs else (None, None))
        print('Load source data. Train shape:',
              source_train.shape, 'Test shape:', source_test.shape)

        # 2. Load target
        target_assembly = self.target_assembly_class()

        try:
            target_train_data = target_assembly.get_assembly(
                **self.target_assembly_train_kwargs)
            if len(target_train_data) == 3:
                target_train, ceiling, stratify_labels_train = target_train_data
            elif len(target_train_data) == 2:
                target_train, ceiling = target_train_data
                stratify_labels_train = None
            else:
                raise ValueError(
                    f"Target assembly returned {len(target_train_data)} values, expected 2 or 3.")
        except Exception as e:
            print(f"Error calling get_assembly for training on target: {e}")
            raise

        target_test = None
        if self.target_assembly_test_kwargs:
            target_test, _ = target_assembly.get_assembly(
                **self.target_assembly_test_kwargs)
            print('Load target data. Train shape:',
                  target_train.shape, 'Test shape:', target_test.shape)

        # 3. Compute metrics
        results = {}
        for name, metric_class in self.metrics.items():
            try:
                extra = self.metric_params.get(name, {})
                if ceiling is not None:
                    metric_instance = metric_class(
                        ceiling=ceiling, **extra)
                else:
                    metric_instance = metric_class(**extra)
                results[name] = metric_instance.compute(
                    source_train,
                    target_train,
                    test_source=source_test,
                    test_target=target_test,
                    stratify_on=stratify_labels_train
                )
            except Exception as e:
                print(f"Metric '{name}' failed and will be skipped: {e}")
                continue

        results['timestamp'] = datetime.datetime.utcnow().isoformat()

        # 4. Save
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'rb') as f:
                    prev = pickle.load(f)
                metrics_list = prev.get("metrics", [])
                if isinstance(metrics_list, dict):
                    metrics_list = [metrics_list]
                metrics_list.append(results)
                merged = {"metrics": metrics_list, "ceiling": ceiling}
            except Exception as e:
                print(
                    f"Could not load existing results file, overwriting: {e}")
                merged = {"metrics": [results], "ceiling": ceiling}
        else:
            merged = {"metrics": [results], "ceiling": ceiling}

        with open(self.results_file, 'wb') as f:
            pickle.dump(merged, f)

        return {'metrics': results, 'ceiling': ceiling}

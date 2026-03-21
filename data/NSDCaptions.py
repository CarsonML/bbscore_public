import json
import os
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

from data.base import BaseDataset
from data.NSDShared import NSDStimulusSet


class NSDCaptionStimulusSet(BaseDataset):
    """
    Dataset for NSD captions aligned to the NSD shared test image set.

    Each sample is a caption string (or its preprocessed representation)
    corresponding to an NSD image index. This is intended to be used as the
    stimulus_train_class in BenchmarkScore for text-based NSD benchmarks.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        captions_path: Optional[str] = None,
        preprocess: Optional[Callable] = None,
    ):
        """
        Initialize the NSDCaptionStimulusSet.

        Args:
            root_dir: Optional root directory (passed to BaseDataset and
                NSDStimulusSet for alignment checks). Does not affect where
                captions are read from.
            captions_path: Path to a JSONL file produced by batch_captions_nsd.py
                with one object per line: {"index": int, "caption": str}.
                If None, we first consult the NSD_CAPTIONS_PATH environment
                variable, then fall back to scanning ./captions/nsd_qwen3vl
                for a single *.jsonl file.
            preprocess: Optional preprocessing callable (typically the model
                preprocess_fn). If provided, __getitem__ will return the
                preprocessed representation; otherwise it will return the raw
                caption string.
        """
        super().__init__(root_dir)
        self.preprocess = preprocess

        # Resolve captions_path if not explicitly provided
        if captions_path is None:
            captions_path = os.environ.get("NSD_CAPTIONS_PATH")

        if captions_path is None:
            default_dir = os.path.join(os.getcwd(), "captions", "nsd_qwen3vl")
            if os.path.isdir(default_dir):
                candidates = [
                    os.path.join(default_dir, f)
                    for f in os.listdir(default_dir)
                    if f.endswith(".jsonl")
                ]
                if len(candidates) == 1:
                    captions_path = candidates[0]

        if captions_path is None or not os.path.isfile(captions_path):
            raise FileNotFoundError(
                "NSDCaptionStimulusSet could not locate a captions JSONL file. "
                "Set NSD_CAPTIONS_PATH to a JSONL produced by batch_captions_nsd.py, "
                "or ensure there is exactly one *.jsonl file in ./captions/nsd_qwen3vl."
            )

        self.captions_path = captions_path
        print(f"NSDCaptionStimulusSet using captions file: {self.captions_path}")
        self._load_captions()
        self._validate_alignment()

    def _load_captions(self) -> None:
        """Load captions from JSONL into an index->caption mapping."""
        self.index_to_caption: Dict[int, str] = {}
        with open(self.captions_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if "index" not in rec or "caption" not in rec:
                    continue
                idx = rec["index"]
                caption = rec["caption"]
                if not isinstance(idx, int):
                    continue
                if not isinstance(caption, str):
                    continue
                # Last write wins if there are duplicates; this matches resume logic.
                self.index_to_caption[idx] = caption

        if not self.index_to_caption:
            raise ValueError(
                f"NSDCaptionStimulusSet found no valid captions in {self.captions_path}"
            )

    def _validate_alignment(self) -> None:
        """
        Ensure that captions are available for every NSD shared test image.

        This uses NSDStimulusSet to determine the expected length and checks
        that indices cover the full range [0, len(NSDStimulusSet)).
        """
        nsd = NSDStimulusSet(root_dir=self.root_dir)
        nsd_len = len(nsd)

        missing = [i for i in range(nsd_len) if i not in self.index_to_caption]
        if missing:
            raise ValueError(
                f"NSDCaptionStimulusSet is missing captions for {len(missing)} "
                f"NSD indices out of {nsd_len}. Example missing indices: "
                f"{missing[:10]}"
            )

        self._length = nsd_len

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for NSDCaptionStimulusSet")
        caption = self.index_to_caption[idx]
        if self.preprocess is None:
            return caption
        return self.preprocess(caption)


class NSDImageCaptionStimulusSet(BaseDataset):
    """
    NSD shared test images paired with captions from a JSONL (same schema as
    NSDCaptionStimulusSet). Each sample is ``{"image": PIL.Image, "caption": str}``
    (or preprocessed output) for multimodal models that take image + text in one forward.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        captions_path: Optional[str] = None,
        preprocess: Optional[Callable] = None,
    ):
        super().__init__(root_dir)
        self.preprocess = preprocess

        if captions_path is None:
            captions_path = os.environ.get("NSD_CAPTIONS_PATH")

        if captions_path is None:
            default_dir = os.path.join(os.getcwd(), "captions", "nsd_qwen3vl")
            if os.path.isdir(default_dir):
                candidates = [
                    os.path.join(default_dir, f)
                    for f in os.listdir(default_dir)
                    if f.endswith(".jsonl")
                ]
                if len(candidates) == 1:
                    captions_path = candidates[0]

        if captions_path is None or not os.path.isfile(captions_path):
            raise FileNotFoundError(
                "NSDImageCaptionStimulusSet could not locate a captions JSONL file. "
                "Set NSD_CAPTIONS_PATH to a JSONL with {\"index\", \"caption\"} per line."
            )

        self.captions_path = captions_path
        print(f"NSDImageCaptionStimulusSet using captions file: {self.captions_path}")
        self.index_to_caption: Dict[int, str] = {}
        self._load_captions()
        self._nsd = NSDStimulusSet(root_dir=root_dir)
        self._nsd._prepare_images()
        nsd_len = len(self._nsd)
        missing = [i for i in range(nsd_len) if i not in self.index_to_caption]
        if missing:
            raise ValueError(
                f"NSDImageCaptionStimulusSet is missing captions for {len(missing)} "
                f"indices out of {nsd_len}. Example missing: {missing[:10]}"
            )
        self._length = nsd_len

    def _load_captions(self) -> None:
        with open(self.captions_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if "index" not in rec or "caption" not in rec:
                    continue
                idx = rec["index"]
                caption = rec["caption"]
                if not isinstance(idx, int):
                    continue
                if not isinstance(caption, str):
                    continue
                self.index_to_caption[idx] = caption

        if not self.index_to_caption:
            raise ValueError(
                f"NSDImageCaptionStimulusSet found no valid captions in {self.captions_path}"
            )

    @staticmethod
    def _truncate_caption(text: str) -> str:
        max_chars = int(os.environ.get("NSD_MULTIMODAL_MAX_CAPTION_CHARS", "32000"))
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for NSDImageCaptionStimulusSet")
        raw = self._nsd.test_image_data[idx]
        if isinstance(raw, np.ndarray):
            pil = Image.fromarray(raw).convert("RGB")
        else:
            pil = raw.convert("RGB") if hasattr(raw, "convert") else Image.fromarray(raw).convert("RGB")
        caption = self._truncate_caption(self.index_to_caption[idx])
        sample = {"image": pil, "caption": caption}
        if self.preprocess is None:
            return sample
        return self.preprocess(sample)


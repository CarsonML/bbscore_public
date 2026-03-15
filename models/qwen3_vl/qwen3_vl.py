import os
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    raise ImportError("Qwen3-VL requires the latest transformers from source. "
                      "Run: pip install git+https://github.com/huggingface/transformers")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class Qwen3VL:
    """Loads pre-trained Qwen3-VL models for feature extraction."""

    def __init__(self):
        """Initializes the Qwen3-VL loader."""
        self.model_mappings = {
            "QWEN3-VL-8B-IMG": "Qwen/Qwen3-VL-8B-Instruct",
            "QWEN3-VL-8B-TXT": "Qwen/Qwen3-VL-8B-Instruct",
            "QWEN3-VL-8B-JOINT": "Qwen/Qwen3-VL-8B-Instruct",
        }

        self.processor = None
        self.mode = None
        self.static = True

    def _determine_mode(self, identifier: str):
        if identifier.endswith("-IMG"):
            return "img"
        elif identifier.endswith("-TXT"):
            return "txt"
        elif identifier.endswith("-JOINT"):
            return "joint"
        return "img"

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for Qwen3-VL using the native apply_chat_template.
        """
        if self.mode != "img":
            raise NotImplementedError(f"Mode {self.mode} preprocessing is not fully implemented yet.")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            img = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            img = Image.fromarray(np.uint8(input_data)).convert("RGB")
        elif isinstance(input_data, Image.Image):
            img = input_data.convert("RGB")
        elif isinstance(input_data, list):
            img = [i.convert("RGB") for i in input_data]
        else:
            raise ValueError("Input must be a PIL Image, file path, or numpy array")

        # Qwen3-VL natively supports putting the image object directly into the messages dict!
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Preparation for inference: apply_chat_template handles images and tokenization now!
        inputs_batch_feature = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Convert HuggingFace BatchFeature into a generic dictionary so standard deep neural extraction unpacking (**inputs) works
        inputs = dict(inputs_batch_feature)

        if torch.cuda.is_available():
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to("cuda")
        
        return inputs

    def get_model(self, identifier):
        identifier = identifier.upper()
        self.mode = self._determine_mode(identifier)

        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Load with Qwen3VLForConditionalGeneration explicitly as per documentation
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                
                # Intercept the forward pass to save image_grid_thw for the postprocess_fn
                original_forward = model.forward
                def forward_with_catch(*args, **kwargs):
                    if "image_grid_thw" in kwargs:
                        self._last_image_grid_thw = kwargs["image_grid_thw"]
                    return original_forward(*args, **kwargs)
                model.forward = forward_with_catch
                
                self.processor = AutoProcessor.from_pretrained(model_name)
                return model

        raise ValueError(f"Unknown model identifier: {identifier}.")

    def postprocess_fn(self, features_np):
        """
        Qwen3-VL vision blocks can emit patch-major activations, so we use
        image_grid_thw to reconstruct a batch-major per-image patch tensor.
        This keeps patch structure intact, matching the token-preserving style
        used by other vision transformers in the repo.
        """
        is_tensor = isinstance(features_np, torch.Tensor)
        expected_batch_size = getattr(self, "_bbscore_expected_batch_size", None)
        current_layer_name = getattr(self, "_bbscore_current_layer_name", "unknown")
        patch_counts = None
        if hasattr(self, '_last_image_grid_thw') and self._last_image_grid_thw is not None:
            patch_counts = self._last_image_grid_thw.prod(dim=1).tolist()

        def stack_patch_major_activations(feats, counts):
            if len(set(counts)) != 1:
                raise ValueError(
                    f"[qwen3_vl] Layer '{current_layer_name}' produced variable patch counts {counts}. "
                    "BBScore currently expects a fixed patch count so batches can be stacked across the dataset."
                )
            patches_per_image = counts[0]
            if is_tensor:
                splits = torch.split(feats, counts, dim=0)
                normalized = []
                for chunk in splits:
                    if chunk.ndim == 3 and chunk.shape[1] == 1:
                        chunk = chunk[:, 0, :]
                    normalized.append(chunk.reshape(patches_per_image, -1))
                return torch.stack(normalized, dim=0)
            indices = np.cumsum(counts)[:-1]
            splits = np.split(feats, indices, axis=0)
            normalized = []
            for chunk in splits:
                if chunk.ndim == 3 and chunk.shape[1] == 1:
                    chunk = chunk[:, 0, :]
                normalized.append(chunk.reshape(patches_per_image, -1))
            return np.stack(normalized, axis=0)
        
        # If the output is truly batch-major [Batch, Sequence, Dim], keep it as-is.
        # Some Qwen3-VL vision blocks instead return patch-major tensors such as
        # [total_patches, 1, hidden], which must be reconstructed via image_grid_thw.
        if features_np.ndim == 3:
            if expected_batch_size is not None and features_np.shape[0] == expected_batch_size:
                return features_np

            if patch_counts is not None:
                print(f"[DEBUG qwen3_vl] 3D tensor shape: {tuple(features_np.shape)}")
                print(f"[DEBUG qwen3_vl] Intercepted image_grid_thw expects batch size {len(patch_counts)}.")
                if expected_batch_size is not None:
                    print(f"[DEBUG qwen3_vl] BBScore batch size was {expected_batch_size}.")
                print(f"[DEBUG qwen3_vl] Patch breakdown per image: {patch_counts}")

                if sum(patch_counts) == features_np.shape[0]:
                    restored = stack_patch_major_activations(features_np, patch_counts)
                    if expected_batch_size is not None and restored.shape[0] != expected_batch_size:
                        raise ValueError(
                            f"[qwen3_vl] Layer '{current_layer_name}' reconstructed {restored.shape[0]} image rows "
                            f"from a patch-major 3D tensor with shape {tuple(features_np.shape)} for a BBScore batch "
                            f"of {expected_batch_size} samples. grid_thw rows={len(patch_counts)}, "
                            f"patch_counts={patch_counts}."
                        )
                    print(f"[DEBUG qwen3_vl] SUCCESS! Returning {restored.shape[0]} per-image patch tensors from 3D activations.")
                    return restored

            if expected_batch_size is not None:
                raise ValueError(
                    f"[qwen3_vl] Layer '{current_layer_name}' produced a 3D tensor with shape {tuple(features_np.shape)}, "
                    f"but BBScore expected {expected_batch_size} samples and no valid patch reconstruction path matched."
                )
            return features_np
        
        # If the output is 2D, it could be [Batch, Dim] (good) or [Sequence, Dim] (needs pooling if batch=1)
        # Qwen3-VL often flattens the batch dimension for image patching.
        if features_np.ndim == 2:
            
            # --- BATCH GRID RECONSTRUCTION ---
            # Qwen3 concatenates image patches across the entire batch natively.
            # E.g. 2 images with 256 and 676 patches become [932, 1536].
            # We must use the intercepted grid to split them properly.
            if patch_counts is not None:
                print(f"[DEBUG qwen3_vl] postprocess_fn received {features_np.shape[0]} total flattened patches.")
                print(f"[DEBUG qwen3_vl] Intercepted image_grid_thw expects batch size {len(patch_counts)}.")
                if expected_batch_size is not None:
                    print(f"[DEBUG qwen3_vl] BBScore batch size was {expected_batch_size}.")
                print(f"[DEBUG qwen3_vl] Patch breakdown per image: {patch_counts}")
                
                if sum(patch_counts) == features_np.shape[0]:
                    restored = stack_patch_major_activations(features_np, patch_counts)
                    if expected_batch_size is not None and restored.shape[0] != expected_batch_size:
                        raise ValueError(
                            f"[qwen3_vl] Layer '{current_layer_name}' reconstructed {restored.shape[0]} image rows "
                            f"from image_grid_thw for a BBScore batch of {expected_batch_size} samples. "
                            f"grid_thw rows={len(patch_counts)}, patch_counts={patch_counts}, "
                            f"raw feature shape={tuple(features_np.shape)}. "
                            f"This suggests one or more inputs expanded into multiple image grids/windows."
                        )
                    print(f"[DEBUG qwen3_vl] SUCCESS! Returning {restored.shape[0]} per-image patch tensors to BBScore.")
                    return restored
                else:
                    print(f"[DEBUG qwen3_vl] WARNING: Grid sum {sum(patch_counts)} != features {features_np.shape[0]}")

            # For a single image without grid metadata, keep the patch sequence intact.
            if expected_batch_size is not None and expected_batch_size != 1:
                raise ValueError(
                    f"[qwen3_vl] Layer '{current_layer_name}' fell back to a single-image patch tensor for a 2D tensor "
                    f"with shape {tuple(features_np.shape)}, but the BBScore batch size was {expected_batch_size}. "
                    f"Missing or incompatible image_grid_thw would collapse multiple samples into one row."
                )
            if is_tensor:
                restored = features_np.unsqueeze(0)
                print(f"[DEBUG qwen3_vl] Fallback executed. Returning single-image patch tensor.")
                return restored
            else:
                restored = np.expand_dims(features_np, axis=0)
                print(f"[DEBUG qwen3_vl] Fallback executed. Returning single-image patch tensor.")
                return restored
            
        return features_np

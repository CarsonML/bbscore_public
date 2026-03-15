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
        Qwen3-VL Vision blocks output sequences of patches (e.g., [256, 1536] per image instead of [1, 256, 1536]).
        For downstream linear probing (Ridge), we need a single vector per image.
        We average pool the sequence dimension to yield [batch_size, feature_dim].
        """
        is_tensor = isinstance(features_np, torch.Tensor)
        expected_batch_size = getattr(self, "_bbscore_expected_batch_size", None)
        current_layer_name = getattr(self, "_bbscore_current_layer_name", "unknown")
        
        # If the output is natively 3D [Batch, Sequence, Dim]
        if features_np.ndim == 3:
            # Average pool across the sequence dimension (dim 1)
            if is_tensor:
                pooled = features_np.mean(dim=1)
            else:
                pooled = np.mean(features_np, axis=1)
            if expected_batch_size is not None and pooled.shape[0] != expected_batch_size:
                raise ValueError(
                    f"[qwen3_vl] Layer '{current_layer_name}' produced {pooled.shape[0]} pooled rows "
                    f"from a 3D tensor with shape {tuple(features_np.shape)}, but the BBScore batch "
                    f"contained {expected_batch_size} samples."
                )
            return pooled
        
        # If the output is 2D, it could be [Batch, Dim] (good) or [Sequence, Dim] (needs pooling if batch=1)
        # Qwen3-VL often flattens the batch dimension for image patching.
        if features_np.ndim == 2:
            
            # --- BATCH GRID RECONSTRUCTION ---
            # Qwen3 concatenates image patches across the entire batch natively.
            # E.g. 2 images with 256 and 676 patches become [932, 1536].
            # We must use the intercepted grid to split them properly.
            if hasattr(self, '_last_image_grid_thw') and self._last_image_grid_thw is not None:
                # get the number of patches for each image in the batch (t * h * w)
                patch_counts = self._last_image_grid_thw.prod(dim=1).tolist()
                
                print(f"[DEBUG qwen3_vl] postprocess_fn received {features_np.shape[0]} total flattened patches.")
                print(f"[DEBUG qwen3_vl] Intercepted image_grid_thw expects batch size {len(patch_counts)}.")
                if expected_batch_size is not None:
                    print(f"[DEBUG qwen3_vl] BBScore batch size was {expected_batch_size}.")
                print(f"[DEBUG qwen3_vl] Patch breakdown per image: {patch_counts}")
                
                if sum(patch_counts) == features_np.shape[0]:
                    if is_tensor:
                        splits = torch.split(features_np, patch_counts, dim=0)
                        pooled = torch.stack([s.mean(dim=0) for s in splits], dim=0)
                        if expected_batch_size is not None and pooled.shape[0] != expected_batch_size:
                            raise ValueError(
                                f"[qwen3_vl] Layer '{current_layer_name}' reconstructed {pooled.shape[0]} pooled rows "
                                f"from image_grid_thw for a BBScore batch of {expected_batch_size} samples. "
                                f"grid_thw rows={len(patch_counts)}, patch_counts={patch_counts}, "
                                f"raw feature shape={tuple(features_np.shape)}. "
                                f"This suggests one or more inputs expanded into multiple image grids/windows."
                            )
                        print(f"[DEBUG qwen3_vl] SUCCESS! Returning {pooled.shape[0]} pooled vectors to BBScore.")
                        return pooled
                    else:
                        indices = np.cumsum(patch_counts)[:-1]
                        splits = np.split(features_np, indices, axis=0)
                        pooled = np.stack([s.mean(axis=0) for s in splits], axis=0)
                        if expected_batch_size is not None and pooled.shape[0] != expected_batch_size:
                            raise ValueError(
                                f"[qwen3_vl] Layer '{current_layer_name}' reconstructed {pooled.shape[0]} pooled rows "
                                f"from image_grid_thw for a BBScore batch of {expected_batch_size} samples. "
                                f"grid_thw rows={len(patch_counts)}, patch_counts={patch_counts}, "
                                f"raw feature shape={tuple(features_np.shape)}. "
                                f"This suggests one or more inputs expanded into multiple image grids/windows."
                            )
                        print(f"[DEBUG qwen3_vl] SUCCESS! Returning {pooled.shape[0]} pooled vectors to BBScore.")
                        return pooled
                else:
                    print(f"[DEBUG qwen3_vl] WARNING: Grid sum {sum(patch_counts)} != features {features_np.shape[0]}")

            # Average pooling the sequence mapping down to a single 1D feature vector for the image
            if expected_batch_size is not None and expected_batch_size != 1:
                raise ValueError(
                    f"[qwen3_vl] Layer '{current_layer_name}' fell back to single-vector pooling for a 2D tensor "
                    f"with shape {tuple(features_np.shape)}, but the BBScore batch size was {expected_batch_size}. "
                    f"Missing or incompatible image_grid_thw would collapse multiple samples into one row."
                )
            if is_tensor:
                pooled = features_np.mean(dim=0).unsqueeze(0)
                print(f"[DEBUG qwen3_vl] Fallback executed. Returning single vector.")
                return pooled
            else:
                pooled = np.expand_dims(np.mean(features_np, axis=0), axis=0)
                print(f"[DEBUG qwen3_vl] Fallback executed. Returning single vector.")
                return pooled
            
        return features_np

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
                
                self.processor = AutoProcessor.from_pretrained(model_name)
                return model

        raise ValueError(f"Unknown model identifier: {identifier}.")

    def postprocess_fn(self, features_np):
        """
        Qwen3-VL Vision blocks output sequences of patches (e.g., [256, 1536] per image instead of [1, 256, 1536]).
        For downstream linear probing (Ridge), we need a single vector per image.
        We average pool the sequence dimension to yield [batch_size, feature_dim].
        """
        # If the output is natively 3D [Batch, Sequence, Dim]
        if features_np.ndim == 3:
            # Average pool across the sequence dimension (dim 1)
            return np.mean(features_np, axis=1)
        
        # If the output is 2D, it could be [Batch, Dim] (good) or [Sequence, Dim] (needs pooling if batch=1)
        # Qwen3-VL often flattens the batch dimension for image patching.
        if features_np.ndim == 2:
            # For BBScore, default extraction passes 1 image at a time natively resulting in [Sequence, Dim]
            # Average pooling the sequence mapping down to a single 1D feature vector for the image
            pooled = np.mean(features_np, axis=0)
            # BBScore extractor expects at least 2D [Batch, Features]
            return np.expand_dims(pooled, axis=0)
            
        return features_np

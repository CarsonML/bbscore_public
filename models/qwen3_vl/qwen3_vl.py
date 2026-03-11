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
        if features_np.ndim == 3:
            batch_size = features_np.shape[0]
            return features_np.reshape(batch_size, -1)
        return features_np

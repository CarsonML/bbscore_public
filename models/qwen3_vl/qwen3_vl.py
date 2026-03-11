import os
import torch
from PIL import Image
import numpy as np

# Qwen3-VL uses the same classes as Qwen2-VL or AutoModelForCausalLM/AutoProcessor
# Qwen3-VL-8B-Instruct is compatible with Qwen2VLForConditionalGeneration or AutoModelForImageTextToText
from transformers import AutoProcessor, AutoModelForCausalLM

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
            # Placeholders for future multimodal toggles
            "QWEN3-VL-8B-TXT": "Qwen/Qwen3-VL-8B-Instruct",
            "QWEN3-VL-8B-JOINT": "Qwen/Qwen3-VL-8B-Instruct",
        }

        self.processor = None
        self.mode = None # "img", "txt", or "joint"
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
        Preprocesses input data for Qwen3-VL.
        For phase 1, we only handle image inputs.

        Args:
            input_data: PIL Image, file path (str), or numpy array.

        Returns:
            Dict[str, torch.Tensor]: Preprocessed inputs for the model.
        """
        # Ensure we are currently in image-only mode 
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

        # Qwen3-VL expects the chat template to format the vision input correctly
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

        # Use the processor to apply the chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        return inputs

    def get_model(self, identifier):
        """
        Loads a Qwen3-VL model based on the identifier.

        Args:
            identifier (str): Identifier for the model variant.

        Returns:
            The loaded model.
        """
        identifier = identifier.upper()
        self.mode = self._determine_mode(identifier)

        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Use AutoModelForImageTextToText to infer the correct Qwen3 architecture type automatically
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                self.processor = AutoProcessor.from_pretrained(model_name)
                
                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available mappings: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """
        Postprocesses model output by flattening features.
        """
        if features_np.ndim == 3:
            # (batch_size, seq_len, feature_dim) -> (batch_size, -1)
            batch_size = features_np.shape[0]
            flattened_features = features_np.reshape(batch_size, -1)
        else:
            flattened_features = features_np
            
        return flattened_features

# Need process_vision_info utility
def process_vision_info(messages):
    """
    Helper function to extract images/videos from the Qwen messages format
    """
    try:
        from qwen_vl_utils import process_vision_info as qvu_process
        return qvu_process(messages)
    except ImportError:
         image_inputs = []
         video_inputs = []
         for msg in messages:
             if isinstance(msg, dict) and 'content' in msg:
                 for c in msg['content']:
                     if isinstance(c, dict):
                         if c.get('type') == 'image' and 'image' in c:
                             image_inputs.append(c['image'])
                         elif c.get('type') == 'video' and 'video' in c:
                             video_inputs.append(c['video'])
         return image_inputs, video_inputs

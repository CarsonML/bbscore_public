import os
import torch
from PIL import Image
import numpy as np

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class Qwen2VL:
    """Loads pre-trained Qwen2-VL models for feature extraction."""

    def __init__(self):
        """Initializes the Qwen2-VL loader."""
        self.model_mappings = {
            "QWEN2-VL-2B-IMG": "Qwen/Qwen2-VL-2B-Instruct",
            # Placeholders for future multimodal toggles
            "QWEN2-VL-2B-TXT": "Qwen/Qwen2-VL-2B-Instruct",
            "QWEN2-VL-2B-JOINT": "Qwen/Qwen2-VL-2B-Instruct",
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
        Preprocesses input data for Qwen2-VL.
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

        # For Qwen2-VL, we must format it as a conversation setup
        # Since this is pure image extraction, we just provide the image with a generic prompt or no prompt if possible.
        # But Qwen2-VL typically expects some text to process the image block properly.
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

        # Move to GPU if available and cast to float16
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # We'll just return the inputs dictionary, as the extraction wrapper will pass it to `model(**inputs)`.
        return inputs

    def get_model(self, identifier):
        """
        Loads a Qwen2-VL model based on the identifier.

        Args:
            identifier (str): Identifier for the model variant.

        Returns:
            The loaded model.
        """
        identifier = identifier.upper()
        self.mode = self._determine_mode(identifier)

        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                # Load the model
                # We use bfloat16 or float16 and auto device map
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Load the processor
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
        # For an LM, we typically get sequence length * feature dim.
        # Since we just want the pooled/flattened representation for BBScore:
        if features_np.ndim == 3:
            # (batch_size, seq_len, feature_dim) -> (batch_size, -1)
            batch_size = features_np.shape[0]
            flattened_features = features_np.reshape(batch_size, -1)
        else:
            flattened_features = features_np
            
        return flattened_features

# Need process_vision_info utility
from transformers.utils import is_vision_available

def process_vision_info(messages):
    """
    Helper function to extract images/videos from the Qwen messages format
    """
    from qwen_vl_utils import process_vision_info as qvu_process
    try:
         image_inputs, video_inputs = qvu_process(messages)
    except ImportError:
         # Fallback if qwen_vl_utils is not installed
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

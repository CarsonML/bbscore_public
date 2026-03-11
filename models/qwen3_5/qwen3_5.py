import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

class Qwen3_5:
    """Loads pre-trained Qwen3.5 models for feature extraction in BBScore."""

    def __init__(self):
        """Initializes the Qwen3.5 loader."""
        self.model_mappings = {
            "QWEN3_5-9B-TXT": "Qwen/Qwen3.5-9B",
            # We map image and joint variants as well for architectural completeness,
            # though it's a text-only model. The inputs must be strings (captions).
            "QWEN3_5-9B-IMG": "Qwen/Qwen3.5-9B",
            "QWEN3_5-9B-JOINT": "Qwen/Qwen3.5-9B",
        }

        self.tokenizer = None
        self.mode = None
        self.static = True

    def _determine_mode(self, identifier: str):
        if identifier.endswith("-IMG"):
            return "img"
        elif identifier.endswith("-TXT"):
            return "txt"
        elif identifier.endswith("-JOINT"):
            return "joint"
        return "txt"

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocesses input data for Qwen3.5-9B.
        Since it is a text-only LLM, it expects string inputs (e.g. captions).

        Args:
            input_data: string, PIL Image, or numpy array.

        Returns:
            Dict[str, torch.Tensor]: Tokenized text input.
        """
        if isinstance(input_data, str) and not os.path.isfile(input_data):
            text = input_data
        elif isinstance(input_data, (Image.Image, np.ndarray)) or (isinstance(input_data, str) and os.path.isfile(input_data)):
            # Fallback if image data is strictly piped into this model
            raise ValueError(
                "Qwen3.5-9B is a text language model and cannot process raw images. "
                "Ensure that your benchmark/dataset wrapper provides text captions instead of image pixels."
            )
        elif isinstance(input_data, dict) and "text" in input_data:
            text = input_data["text"]
        else:
            text = "Describe the visual scene corresponding to this underlying data."

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]

        # Use the chat template configured on Qwen tokenizer
        text_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text_formatted],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        return inputs

    def get_model(self, identifier):
        """
        Loads a Qwen3.5 model based on the identifier.
        """
        identifier = identifier.upper()
        self.mode = self._determine_mode(identifier)

        for prefix, model_name in self.model_mappings.items():
            if identifier.startswith(prefix):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Qwen3.5 may not define a pad_token by default
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                return model

        raise ValueError(
            f"Unknown model identifier: {identifier}. "
            f"Available mappings: {', '.join(self.model_mappings.keys())}"
        )

    def postprocess_fn(self, features_np):
        """
        Postprocesses model output by flattening features into a feature vector.
        """
        if features_np.ndim == 3:
            batch_size = features_np.shape[0]
            flattened_features = features_np.reshape(batch_size, -1)
        else:
            flattened_features = features_np
            
        return flattened_features

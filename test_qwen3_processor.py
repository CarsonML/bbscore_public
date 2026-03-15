from transformers import AutoProcessor
from PIL import Image
import torch
processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')
img = Image.new('RGB', (224, 224))
messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "test"}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
print(type(inputs))
for k, v in dict(inputs).items():
    print(k, type(v), getattr(v, "shape", None))

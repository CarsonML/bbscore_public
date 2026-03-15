from transformers import AutoProcessor
from PIL import Image
import torch

def main():
    processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')
    img = Image.new('RGB', (224, 224))
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "test"}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    
    print("Type of inputs:", type(inputs))
    for k, v in dict(inputs).items():
        print("Key:", k, "| Type:", type(v))
        if isinstance(v, torch.Tensor):
            print("  Shape:", v.shape)
        elif isinstance(v, list):
            print("  List len:", len(v), "| First element type:", type(v[0]) if len(v) > 0 else "empty")

if __name__ == "__main__":
    main()

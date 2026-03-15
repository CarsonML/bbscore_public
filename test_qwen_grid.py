import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    pass
from PIL import Image

def main():
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        print("Model load failed:", e)
        return

    img1 = Image.new('RGB', (224, 224))
    img2 = Image.new('RGB', (425, 425)) # different size image
    
    messages = [
        {"role": "user", "content": [{"type": "image", "image": img1}, {"type": "image", "image": img2}, {"type": "text", "text": "test"}]}
    ]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    
    print("image_grid_thw:", inputs["image_grid_thw"])
    print("pixel_values shape:", inputs["pixel_values"].shape)
    
if __name__ == "__main__":
    main()

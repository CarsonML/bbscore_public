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
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")
    except Exception as e:
        print("Model load failed:", e)
        return

    img = Image.new('RGB', (224, 224))
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe this image."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in dict(inputs).items() if isinstance(v, torch.Tensor)}
    
    with torch.inference_mode():
        # First test the bare vision encoder to see what shape it outputs given pixel tensors natively
        vision_output = model.visual(inputs['pixel_values'], inputs['image_grid_thw'])
        print(f"Vision Encoder full output shape: {vision_output.shape}")
        
    print("Done")

if __name__ == "__main__":
    main()

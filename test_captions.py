import argparse
import random
import torch
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from data.NSDShared import NSDStimulusSet

# ImageNet un-normalization constants (from dataset transforms)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnormalize_to_pil(tensor):
    """Reverts the PyTorch normalization so the image can be viewed correctly."""
    tensor = tensor.cpu()
    tensor = tensor * STD + MEAN
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)
    array = np.transpose(array, (1, 2, 0)) # CHW -> HWC
    return Image.fromarray(array)

def img_to_base64(pil_img):
    """Converts a PIL image to a base64 string for embedding in single-file HTML."""
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL captioning on random NSD images")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to feed to Qwen3-VL along with the image")
    parser.add_argument("--num_images", type=int, default=10, help="Number of random images to test (default: 10)")
    parser.add_argument("--output", type=str, default="review_captions.html", help="HTML output path")
    args = parser.parse_args()

    print("Loading NSD Dataset...")
    dataset = NSDStimulusSet()
    
    # Pick N random image indices
    num_samples = min(args.num_images, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    print("Loading Qwen3-VL-8B-Instruct...")
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    print(f"\nStarting generation for {num_samples} images Using Prompt: '{args.prompt}'\n")

    # Start constructing our single-file standalone HTML document
    html_content = [
        "<html><head><title>Qwen3-VL Caption Review</title><style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: auto; padding: 20px; background-color: #f0f2f5; }",
        ".header { text-align: center; margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}",
        ".item { display: flex; flex-direction: column; align-items: center; margin-bottom: 30px; background-color: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "img { max-width: 400px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #ddd; }",
        "p { font-size: 16px; background-color: #f8f9fa; padding: 15px; border-radius: 8px; width: 90%; border-left: 4px solid #007bff; line-height: 1.5;}",
        "</style></head><body>",
        f"<div class='header'><h2>Caption Review</h2><p><strong>Prompt used:</strong> \"{args.prompt}\"</p></div>"
    ]

    for i, idx in enumerate(indices):
        print(f"[{i+1}/{num_samples}] Extracting and generating for NSD Image Index: {idx}...")
        
        # Pull tensor from dataset and convert to human viewable PIL
        tensor_img = dataset[idx]
        pil_img = unnormalize_to_pil(tensor_img)
        
        # Build the exact chat format required by Qwen3
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_img,
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate text tokens
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        
        # Trim input tokens from output tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode to text
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Print to terminal
        print(f"--> Caption: {output_text}\n")

        # Encode image to base64 and append to HTML
        b64_img = img_to_base64(pil_img)
        html_content.append("<div class='item'>")
        html_content.append(f"<img src='data:image/jpeg;base64,{b64_img}' />")
        html_content.append(f"<p><strong>Qwen3-VL:</strong> {output_text}</p>")
        html_content.append("</div>")

    html_content.append("</body></html>")

    # Save HTML to disk
    with open(args.output, "w") as f:
        f.write("\n".join(html_content))
    
    print("\n" + "="*80)
    print(f"DONE! Visual review file written to: {os.path.abspath(args.output)}")
    print(f"To view the images and their generated text together:")
    print(f"1. Download '{args.output}' to your local machine using VSCode, SCP, or 'cat' and copy-pasting.")
    print("2. Double-click the HTML file to open it in your browser!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

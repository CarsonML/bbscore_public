from models.qwen3_vl.qwen3_vl import Qwen3VL
from PIL import Image
import torch
import numpy as np

m = Qwen3VL()
m.get_model("qwen3_vl_8b_img")

img = Image.new('RGB', (224, 224))
inputs = m.preprocess_fn(img)
print("Keys:", inputs.keys())
print("Pixel shape:", inputs["pixel_values"].shape)

with torch.inference_mode():
    out = m.model.visual(inputs["pixel_values"].to(m.model.device), grid_thw=inputs["image_grid_thw"].to(m.model.device))
    print("Raw visual output shape:", out.shape)
    
    # Simulate BBScore extraction gathering
    # Usually it gathers out.cpu().numpy() into a list then concatenates it
    features = [out.cpu().numpy()]
    gathered = np.concatenate(features, axis=0)
    print("Gathered shape:", gathered.shape)
    
    postprocessed = m.postprocess_fn(gathered)
    print("Postprocessed shape:", postprocessed.shape)

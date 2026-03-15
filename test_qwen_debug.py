import torch
import numpy as np
from PIL import Image
from models.qwen3_vl.qwen3_vl import Qwen3VL

print("Testing Model Hook")
m = Qwen3VL()
# fake grid for batch of 2
m._last_image_grid_thw = torch.tensor([[1,2,2], [1,3,3]]) 

print("\n--- Testing Torch ---")
# 4 patches + 9 patches = 13 patches total
fake_features = torch.randn(13, 20)
out = m.postprocess_fn(fake_features)
print(f"Final out shape: {out.shape}")

print("\n--- Testing Numpy ---")
fake_features_np = np.random.randn(13, 20)
out_np = m.postprocess_fn(fake_features_np)
print(f"Final out_np shape: {out_np.shape}")


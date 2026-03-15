from models.qwen3_vl.qwen3_vl import Qwen3VL
from PIL import Image
import torch

m = Qwen3VL()
m.get_model("QWEN3-VL-8B-IMG")

img1 = Image.new('RGB', (224, 224))
img2 = Image.new('RGB', (425, 425))

inputs1 = m.preprocess_fn(img1)
inputs2 = m.preprocess_fn(img2)

# Simulate batching
# In BBScore, custom_collate groups lists of tensors and extractor applies torch.cat(..., dim=0)
batch_inputs = {
    "pixel_values": torch.cat([inputs1["pixel_values"], inputs2["pixel_values"]], dim=0),
    "image_grid_thw": torch.cat([inputs1["image_grid_thw"], inputs2["image_grid_thw"]], dim=0)
}
print("Batched Patches shape:", batch_inputs["pixel_values"].shape)
print("Batched Grid:", batch_inputs["image_grid_thw"])

with torch.inference_mode():
    out = m.model.visual(**batch_inputs)
    print("Raw visual output shape:", out.shape)
    
    # BBScore extractor_wrapper gets 'features_val' exactly like out
    # but first it has to trigger the monkey-patched self.model_instance._last_image_grid_thw!
    # Let's call model.forward to trigger the interception
    # Note: the extractor runs the full forward pass
    dummy = {"input_ids": torch.tensor([[1, 2], [3, 4]], device=m.model.device), "pixel_values": batch_inputs["pixel_values"].to(m.model.device), "image_grid_thw": batch_inputs["image_grid_thw"].to(m.model.device)}
    
    try:
        m.model(**dummy, output_hidden_states=True)
    except Exception as e:
        # Ignore generation errors, we just want to hit the forward monkey patch
        pass
    
    # Fake the features hitting postprocess_fn
    # Note: we need to manually set the intercepted grid for this test since dummy might have failed
    m._last_image_grid_thw = batch_inputs["image_grid_thw"]
    
    postprocessed = m.postprocess_fn(out)
    print("Postprocessed shape:", postprocessed.shape)

import torch
from models.qwen3_vl.qwen3_vl import Qwen3VL
from PIL import Image

m = Qwen3VL()
m.get_model("QWEN3-VL-8B-IMG")

# Make 3 different sized images
img1 = Image.new('RGB', (224, 224))
img2 = Image.new('RGB', (425, 425))
img3 = Image.new('RGB', (100, 300))

# Preprocess individually
in1 = m.preprocess_fn(img1)
in2 = m.preprocess_fn(img2)
in3 = m.preprocess_fn(img3)

print("Image 1 grid:", in1['image_grid_thw'])
print("Image 2 grid:", in2['image_grid_thw'])
print("Image 3 grid:", in3['image_grid_thw'])

# Mock collate - this is what BBScore's custom_collate docs say it does:
# Group tensors into lists
batch = {
    "pixel_values": [in1["pixel_values"], in2["pixel_values"], in3["pixel_values"]],
    "image_grid_thw": [in1["image_grid_thw"], in2["image_grid_thw"], in3["image_grid_thw"]]
}

# This is what extractor_wrapper._set_weights does (for rank>1 it concatenates)
# Extractor concat:
pixel_values = torch.cat(batch["pixel_values"], dim=0)
image_grid_thw = torch.cat(batch["image_grid_thw"], dim=0)

print(f"Collated pixel_values: {pixel_values.shape}")
print(f"Collated grid: {image_grid_thw}")

# Mock forward
with torch.inference_mode():
    try:
        # Trigger the monkeypatch
        m.model(pixel_values=pixel_values.cuda(), image_grid_thw=image_grid_thw.cuda(), input_ids=torch.tensor([[1]]).cuda())
    except:
        pass

    # The actual vision encoder call:
    vision_out = m.model.visual(pixel_values.cuda(), grid_thw=image_grid_thw.cuda())
    print("Vision encoder out shape:", vision_out.shape)

    # Note that intercept uses kwargs. We need to manually set it for the mock:
    m._last_image_grid_thw = image_grid_thw.cuda()
    
    # Mock extractor wrapping it in a list before postprocess_fn
    # The extractor_wrapper does this:
    features_val = [vision_out]
    processed_features = []
    for feat in features_val:
        processed_features.append(feat)
    
    if len(processed_features) == 1:
        features_val_to_pp = processed_features[0]
    
    print("Shape before postprocess:", features_val_to_pp.shape)
    
    postprocessed = m.postprocess_fn(features_val_to_pp)
    print("Shape AFTER postprocess:", postprocessed.shape)


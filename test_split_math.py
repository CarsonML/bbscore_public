import torch

# Simulate what Qwen3 outputs for a batch of 3 images
# img1 is 2x2 patches (Total: 4)
# img2 is 1x4 patches (Total: 4)
# img3 is 3x3 patches (Total: 9)

# Grid format is [Depth, Height, Width]
image_grid_thw = torch.tensor([
    [1, 2, 2],
    [1, 1, 4],
    [1, 3, 3]
])

# Generate a fake array of patches for the whole batch
# Total patches = 4 + 4 + 9 = 17
total_patches = 17
features_np = torch.randn(total_patches, 1536) # Sequence x Dim

# Simulate postprocess_fn interception
patch_counts = image_grid_thw.prod(dim=1).tolist()
print("Patch counts per image:", patch_counts)

if sum(patch_counts) == features_np.shape[0]:
    splits = torch.split(features_np, patch_counts, dim=0)
    print("Number of splits (Should be 3):", len(splits))
    
    # Average pool each image sequence
    final_output = torch.stack([s.mean(dim=0) for s in splits], dim=0)
    print("Final Output Shape (Should be [3, 1536]):", final_output.shape)
else:
    print("Mismatch!", sum(patch_counts), features_np.shape[0])


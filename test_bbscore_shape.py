import pickle
import numpy as np

features_file = "/home/users/carsonml/scikit_learn_data/features/qwen3_vl_8b_img_model.visual.blocks.26_NSDV1Shared_features.pkl"
try:
    with open(features_file, 'rb') as f:
        data = pickle.load(f)
    print("X shape:", data['train'].shape)
    print("y shape:", len(data['train_labels']))
except Exception as e:
    print("Could not load features:", e)


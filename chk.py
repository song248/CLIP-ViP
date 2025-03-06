import numpy as np

feature_path = "output/features.npy"
features = np.load(feature_path)

print(f"Feature shape: {features.shape}")  # (batch_size, feature_dim)
print(f"Feature dtype: {features.dtype}")  # float32, float16
print(f"First 5 feature vectors:\n{features[:5]}")

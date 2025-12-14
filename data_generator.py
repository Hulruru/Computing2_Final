import numpy as np
from sklearn.datasets import make_blobs
import random
import os

NUM_FILES = 30
N_SAMPLES = 10000000
N_FEATURES = 100
N_CLUSTERS = 10
DATA_DIR = "dataset"

os.makedirs(DATA_DIR, exist_ok=True)
print(f"Generating {NUM_FILES} datasets in '{DATA_DIR}/'...")
print(f"{'File':<15} | {'Type':<10} | {'Target %':<10} | {'Actual Sparsity':<15}")
print("-" * 60)

for i in range(NUM_FILES):
    X, _ = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLUSTERS, random_state=42 + i)
    X = X.astype(np.float32)

    is_sparse_target = random.random() > 0.5 

    if is_sparse_target:
        target_sparsity = random.uniform(0.80, 0.95)
        type_label = "Sparse"
    else:
        target_sparsity = random.uniform(0.00, 0.20)
        type_label = "Dense"

    mask = np.random.rand(N_SAMPLES, N_FEATURES) < target_sparsity
    X[mask] = 0.0

    filename = os.path.join(DATA_DIR, f"data_{i}.bin")
    X.tofile(filename)

    actual_sparsity = 1.0 - (np.count_nonzero(X) / X.size)
    
    print(f"data_{i}.bin         | {type_label:<10} | {target_sparsity*100:.1f}%      | {actual_sparsity:.2%}")

print("-" * 60)
print("Done. All binary files generated.")
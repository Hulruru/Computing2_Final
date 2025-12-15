import numpy as np
from sklearn.datasets import make_blobs
import random
import os

# ==========================================
# Configuration
# ==========================================
NUM_FILES = 30          # 생성할 파일 개수
N_SAMPLES = 10000000    # 샘플 개수 (10M)
N_FEATURES = 100        # 차원 수 (Feature)
N_CLUSTERS = 10         # 클러스터 개수 (Blob Center)
DATA_DIR = "dataset"    # 저장 경로

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Generating {NUM_FILES} datasets in '{DATA_DIR}/'...")
    print(f"Configuration: {N_SAMPLES} samples x {N_FEATURES} features (Float32)")
    print("-" * 75)
    print(f"{'File':<15} | {'Type':<10} | {'Target %':<10} | {'Actual Sparsity':<15}")
    print("-" * 75)

    for i in range(NUM_FILES):
        # 1. 기본 클러스터 데이터 생성 (Dense)
        # random_state를 변경하여 파일마다 다른 분포 생성
        X, _ = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLUSTERS, random_state=42 + i)
        X = X.astype(np.float32)

        # 2. 희소도(Sparsity) 설정
        # 50% 확률로 Sparse(CSR용) 또는 Dense(SoA용) 데이터 생성
        is_sparse_target = random.random() > 0.5 

        if is_sparse_target:
            # [Sparse Case] Sparsity > 80% (CSR Kernel Trigger)
            target_sparsity = random.uniform(0.80, 0.95)
            type_label = "Sparse"
        else:
            # [Dense Case] Sparsity < 20% (Dense SoA Kernel Trigger)
            target_sparsity = random.uniform(0.00, 0.20)
            type_label = "Dense"

        # 3. 0값 주입 (Masking)
        mask = np.random.rand(N_SAMPLES, N_FEATURES) < target_sparsity
        X[mask] = 0.0

        # 4. 바이너리 파일 저장 (C++ load_bin 함수 호환)
        filename = os.path.join(DATA_DIR, f"data_{i}.bin")
        X.tofile(filename)

        # 5. 실제 희소도 검증 및 로그 출력
        actual_sparsity = 1.0 - (np.count_nonzero(X) / X.size)
        
        print(f"data_{i}.bin        | {type_label:<10} | {target_sparsity*100:.1f}%      | {actual_sparsity:.2%}")

    print("-" * 75)
    print("Done. All binary files generated.")
import numpy as np
import cuml
from cuml.cluster import KMeans
import cupy as cp 
import os
import time
import sys

# ==========================================
# 설정 및 상수
# ==========================================
DATA_DIR = "dataset"
N_FEATURES = 100
N_CLUSTERS = 10
MAX_ITER = 10

def load_binary_data(filename, n_features):
    """바이너리 파일에서 데이터를 로드하여 NumPy 배열로 반환"""
    try:
        data = np.fromfile(filename, dtype=np.float32)
    except FileNotFoundError:
        return None
        
    if data.size % n_features != 0:
        print(f"Error: File size not divisible by {n_features}")
        return None
        
    n_samples = data.size // n_features
    data = data.reshape(n_samples, n_features)
    return data

def run_benchmark(start_idx, end_idx):
    print(f"=== [Python cuML K-Means Benchmark (Pure Compute Time, Files {start_idx} to {end_idx})] ===")
    
    # 1. GPU Warm-up
    # 첫 실행 시 발생하는 CUDA Context 초기화 및 라이브러리 로딩 시간을 측정에서 배제
    print("Performing Warm-up...", end=" ")
    dummy_data = cp.random.rand(1000, N_FEATURES, dtype=np.float32)
    warmup_kmeans = KMeans(n_clusters=N_CLUSTERS, max_iter=1, init='random')
    warmup_kmeans.fit(dummy_data)
    cp.cuda.Stream.null.synchronize() # CUDA 동기화
    print("Done.\n")
    
    total_time = 0.0
    success_count = 0

    file_idx = start_idx
    for file_idx in range(start_idx, end_idx + 1):
        filename = os.path.join(DATA_DIR, f"data_{file_idx}.bin")
        
        if not os.path.exists(filename):
            print(f"\n--- WARNING: File {filename} not found. Stopping batch. ---")
            break 

        print(f"Processing File [{file_idx}]: {filename}")
        
        # 2. 데이터 로드 (CPU)
        data_cpu = load_binary_data(filename, N_FEATURES)
        if data_cpu is None: break
        
        n_samples = data_cpu.shape[0]
        
        # 희소도 확인 (참고용 로그)
        zero_count = np.sum(np.abs(data_cpu) < 1e-6)
        sparsity = zero_count / data_cpu.size
        print(f"    -> Samples: {n_samples}, Sparsity: {sparsity*100:.2f}%")

        # 3. 데이터 전송 (CPU -> GPU) 벤치마크 시간 측정에서 제외 (알고리즘 순수 성능 비교 목적)
        data_gpu = cp.array(data_cpu)

        # cuML 모델 설정
        kmeans = KMeans(n_clusters=N_CLUSTERS, 
                        max_iter=MAX_ITER, 
                        init='random', 
                        output_type='numpy') 

        # 4. 성능 측정 (Pure Compute Time)
        cp.cuda.Stream.null.synchronize() # 이전 작업 완료 대기
        
        start_time = time.perf_counter()
        kmeans.fit(data_gpu)              # K-Means 실행
        cp.cuda.Stream.null.synchronize() # 커널 실행 완료 대기 (필수)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        print(f"    -> Pure Compute Time: {elapsed_ms:.4f} ms")
        
        total_time += elapsed_ms
        success_count += 1
    
    # 결과 요약 출력
    if success_count > 0:
        last_processed_idx = file_idx if file_idx <= end_idx and os.path.exists(os.path.join(DATA_DIR, f"data_{file_idx}.bin")) else file_idx - 1

        print(f"\n================================================")
        print(f"Batch Processing Final Summary")
        print(f"Files Processed: {start_idx} to {last_processed_idx} ({success_count} total)")
        print(f"Overall Average Pure Compute Time: {total_time / success_count:.4f} ms")
        print(f"================================================")
    else:
           print(f"\n================================================")
           print(f"No files were processed in the range {start_idx} to {end_idx}.")
           print(f"================================================")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            start_index = int(sys.argv[1])
            end_index = int(sys.argv[2])
            
            if start_index < 0 or end_index < start_index:
                raise ValueError("Invalid range. Start index must be non-negative and less than or equal to end index.")
                
            run_benchmark(start_index, end_index)
            
        except ValueError as e:
            print(f"Error: Invalid arguments. Both arguments must be integers. {e}")
            print(f"Usage: python3 <script_name>.py <start_file_index> <end_file_index>")
            sys.exit(1)
            
    else:
        print("Error: Two integer arguments are required: <start_file_index> and <end_file_index>.")
        print("Example: python3 cumltest.py 0 4 (Processes data_0.bin to data_4.bin)")
        sys.exit(1)
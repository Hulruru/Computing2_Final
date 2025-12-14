import time
import numpy as np
import os
import glob
from sklearn.cluster import KMeans

def run_cpu_kmeans_batch(folder_path='dataset', output_folder='cpu_results', n_features=100, n_clusters=10):
    file_pattern = os.path.join(folder_path, "data_*.bin")
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"Error: No .bin files found in '{folder_path}'. Please run gen_data.py first.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    print(f"=== [CPU] Scikit-learn KMeans Batch Processing ({len(files)} files) ===")
    print(f"Results will be saved to: ./{output_folder}/")
    
    execution_times = []

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        file_idx = filename.replace("data_", "").replace(".bin", "")
        
        print(f"\nProcessing [{i+1}/{len(files)}]: {filename} ...")

        try:
            X = np.fromfile(filepath, dtype=np.float32)
            X = X.reshape(-1, n_features)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        n_samples = X.shape[0]
        
        sparsity = 1.0 - (np.count_nonzero(X) / X.size)
        print(f"   -> Shape: {X.shape} | Sparsity: {sparsity:.2%}")

        init_centroids = X[:n_clusters] 
        kmeans = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1)

        start_fit = time.time()
        kmeans.fit(X)
        end_fit = time.time()
        
        duration_ms = (end_fit - start_fit) * 1000
        execution_times.append(duration_ms)

        print(f"   -> Execution Time: {duration_ms:.4f} ms")

        labels_path = os.path.join(output_folder, f"labels_{file_idx}.txt")
        np.savetxt(labels_path, kmeans.labels_, fmt='%d')
        
        centroids_path = os.path.join(output_folder, f"centroids_{file_idx}.txt")
        np.savetxt(centroids_path, kmeans.cluster_centers_, fmt='%.6f')
        
        print(f"   -> Saved: {labels_path}, {centroids_path}")

    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print("\n========================================")
        print(f"Batch Benchmark Finished.")
        print(f"Total Files Processed: {len(execution_times)}")
        print(f"Average Execution Time: {avg_time:.4f} ms")
        print("========================================")

if __name__ == "__main__":
    run_cpu_kmeans_batch(folder_path='dataset', output_folder='cpu_results')
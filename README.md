\# Batch Adaptive CUDA K-Means



\*\*High-Performance Parallel K-Means Clustering with Adaptive Kernel Selection\*\*



This project is a \*\*CUDA-based parallel K-Means clustering implementation\*\* designed for processing large-scale datasets ($10^7$ Samples, 100 Features). It adopts an \*\*Adaptive Strategy\*\* that dynamically selects the optimal kernel between \*\*Dense (SoA)\*\* mode and \*\*Sparse (CSR)\*\* mode by analyzing data sparsity in real-time.



We achieved over \*\*50x speedup\*\* compared to CPU-based libraries (`scikit-learn`) and \*\*over 2x performance improvement\*\* compared to the state-of-the-art GPU library (`cuML`).



---



\## 🚀 Key Features



\### 1. Adaptive Kernel Selection

\* Automatically determines the execution mode by analyzing \*\*Sparsity\*\* upon data loading.

&nbsp;   \* \*\*Dense Mode (Sparsity $\\le$ 80%):\*\* Executes logic optimized for memory coalesced access using SoA layout.

&nbsp;   \* \*\*Sparse Mode (Sparsity > 80%):\*\* Executes CSR-based logic to skip unnecessary zero-value computations.



\### 2. Advanced CUDA Optimizations

\* \*\*Memory Coalescing (AoS $\\to$ SoA):\*\* Transposes dense data layout to maximize GPU global memory bandwidth.

\* \*\*Constant Memory:\*\* Caches read-only Centroids data in `\_\_constant\_\_` memory for every iteration to reduce lookup latency.

\* \*\*Shared Memory Reduction:\*\* Computes partial sums within thread blocks first to minimize contention during Global Atomic operations.



\### 3. High Precision \& Verification

\* \*\*Mixed Precision:\*\* Performs distance calculations in `float` for speed, but uses `double` Atomic operations for the Centroid Update step to minimize floating-point errors.

\* \*\*Validation:\*\* Verifies algorithm correctness by comparing MSE (Mean Squared Error) against `scikit-learn` CPU results.



---



\## 📊 Performance Benchmark



\* \*\*Dataset:\*\* 10M Samples, 100 Features, 10 Clusters (Synthetic Data)

\* \*\*Hardware:\*\* NVIDIA GeForce RTX 3090 (24GB), CUDA 12.4



| Implementation | Execution Time (Avg) | Speedup (vs CPU) | Note |

| :--- | :--- | :--- | :--- |

| \*\*Scikit-learn (CPU)\*\* | 24,556.43 ms | 1.00x | Intel Xeon CPU |

| \*\*cuML (GPU Lib)\*\* | 993.25 ms | 24.72x | RAPIDS Library |

| \*\*Naive CUDA\*\* | 3,650.78 ms | 6.73x | Baseline (AoS) |

| \*\*Optimized CUDA\*\* | 1,679.87 ms | 14.62x | SoA + Shared Mem |

| \*\*Adaptive CUDA (Ours)\*\* | \*\*462.45 ms\*\* | \*\*53.10x\*\* | \*\*CSR + Adaptive\*\* |



> \*\*Result:\*\* Our Adaptive implementation demonstrated \*\*~2.1x faster performance\*\* than the commercial `cuML` library.



---



\## 🛠️ System Requirements



\* \*\*OS:\*\* Linux (Ubuntu 20.04+ recommended)

\* \*\*GPU:\*\* NVIDIA GPU (Compute Capability 8.0+ recommended, Tested on RTX 3090)

\* \*\*Compiler:\*\* `nvcc` (CUDA Toolkit 12.4+), `g++`

\* \*\*Python:\*\* 3.10 ~ 3.12 (for Data Gen \& Benchmarking)



---



\## 📦 Installation \& Setup



\### 1. Environment Setup (Recommended)

We strongly recommend using a Conda environment for compatibility with RAPIDS (`cuml`).



```bash

\# Create Conda Environment (Python 3.11 \& CUDA 12.4)

conda create -n rapids\_env -c rapidsai -c conda-forge -c nvidia \\

&nbsp;   cuml=24.02 python=3.11 cuda-version=12.4 numpy scikit-learn



\# Activate Environment

conda activate rapids\_env



\# run data\_generator.py

python3 data\_generator.py



\# run run\_kmeans.sh

chmod +x run\_all\_kmeans.sh

./run\_kmeans.sh



\# if you want to compile and run each cu file

nvcc -o kmeans mCSRKmeans.cu -O3 -arch=sm\_86

./mCSRKmeans


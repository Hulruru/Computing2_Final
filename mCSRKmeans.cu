#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cuda_runtime.h>

#define N_CLUSTERS 10
#define MAX_FEATURES 100    
#define MAX_ITER 10
#define THREADS_PER_BLOCK 256
#define SPARSITY_THRESHOLD 0.8f

__constant__ float c_centroids[N_CLUSTERS * MAX_FEATURES];
__constant__ float c_centroid_norms[N_CLUSTERS];

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void assignClusterKernel_SoA(const float* __restrict__ d_data_soa, int* d_labels, int n_samples, int n_features, int n_clusters) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_samples) {
        float min_dist = FLT_MAX;
        int best_cluster = 0;
        for (int c = 0; c < n_clusters; ++c) {
            float current_dist = 0.0f;
            for (int f = 0; f < n_features; ++f) {
                float val = d_data_soa[f * n_samples + gid];
                float diff = val - c_centroids[c * n_features + f];
                current_dist += diff * diff;
            }
            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = c;
            }
        }
        d_labels[gid] = best_cluster;
    }
}

__global__ void computeNewCentroidsKernel_Shared(
    const float* __restrict__ d_data_soa, 
    const int* __restrict__ d_labels, 
    double* d_new_centroids, 
    int* d_cluster_counts, 
    int n_samples, int n_features, int n_clusters) 
{
    extern __shared__ float s_block_centroids[]; 
    int* s_block_counts = (int*)&s_block_centroids[n_clusters * n_features];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_clusters * n_features;

    for (int i = tid; i < total_elements; i += blockDim.x) s_block_centroids[i] = 0.0f;
    for (int i = tid; i < n_clusters; i += blockDim.x) s_block_counts[i] = 0;
    __syncthreads();

    if (gid < n_samples) {
        int cluster_id = d_labels[gid];
        atomicAdd(&s_block_counts[cluster_id], 1);
        for (int f = 0; f < n_features; ++f) {
            float val = d_data_soa[f * n_samples + gid];
            atomicAdd(&s_block_centroids[cluster_id * n_features + f], val); 
        }
    }
    __syncthreads();

    for (int i = tid; i < n_clusters; i += blockDim.x) {
        if (s_block_counts[i] > 0) atomicAdd(&d_cluster_counts[i], s_block_counts[i]);
    }
    for (int i = tid; i < total_elements; i += blockDim.x) {
        float sum = s_block_centroids[i];
        if (abs(sum) > 1e-6)
            atomicAdd(&d_new_centroids[i], (double)sum);
    }
}

__global__ void assignClusterKernel_CSR(
    const float* __restrict__ csr_values,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ csr_row_ptr,
    const float* __restrict__ data_norms, 
    int* d_labels, 
    int n_samples) 
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_samples) {
        int start_idx = csr_row_ptr[gid];
        int end_idx = csr_row_ptr[gid + 1];
        
        float my_norm = data_norms[gid];
        float min_dist = FLT_MAX;
        int best_cluster = 0;

        for (int c = 0; c < N_CLUSTERS; ++c) {
            float dot_product = 0.0f;
            for (int i = start_idx; i < end_idx; ++i) {
                int col = csr_col_ind[i];
                float val = csr_values[i];
                dot_product += val * c_centroids[c * MAX_FEATURES + col];
            }
            float dist = my_norm + c_centroid_norms[c] - (2.0f * dot_product);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        d_labels[gid] = best_cluster;
    }
}

__global__ void computeNewCentroidsKernel_CSR(
    const float* __restrict__ csr_values,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ d_labels,
    double* d_new_centroids,
    int* d_cluster_counts,
    int n_samples)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_samples) {
        int cluster_id = d_labels[gid];
        atomicAdd(&d_cluster_counts[cluster_id], 1);

        int start_idx = csr_row_ptr[gid];
        int end_idx = csr_row_ptr[gid + 1];

        for (int i = start_idx; i < end_idx; ++i) {
            int col = csr_col_ind[i];
            float val = csr_values[i];
            atomicAdd(&d_new_centroids[cluster_id * MAX_FEATURES + col], (double)val);
        }
    }
}

__global__ void averageCentroidsKernel(float* d_centroids, double* d_new_centroids, int* d_cluster_counts, int n_features, int n_clusters) {
    int cid = threadIdx.x; 
    if (cid < n_clusters) {
        int count = d_cluster_counts[cid];
        if (count > 0) {
            for (int f = 0; f < n_features; ++f) {
                d_centroids[cid * n_features + f] = (float)(d_new_centroids[cid * n_features + f] / count);
            }
        }
    }
}

bool load_bin(const std::string& filename, std::vector<float>& data, int& n_samples, int n_features) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size == 0 || size % sizeof(float) != 0) return false;
    size_t total_elements = size / sizeof(float);
    if (total_elements % n_features != 0) {
        std::cerr << "Error: Total elements not divisible by features." << std::endl;
        exit(1);
    }
    n_samples = total_elements / n_features;
    data.resize(total_elements);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Error: Failed to read data." << std::endl;
        exit(1);
    }
    file.close();
    return true;
}

bool read_cpu_centroids(const std::string& filename, std::vector<float>& centroids, int n_features) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    float val;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        while (ss >> val) centroids.push_back(val);
    }
    file.close();
    return centroids.size() == (size_t)(N_CLUSTERS * n_features);
}

float calculate_centroid_mse(const std::vector<float>& cuda_centroids, const std::vector<float>& cpu_centroids, int n_features) {
    if (cpu_centroids.empty()) return -1.0f;
    double total_mse = 0.0;
    for (int i = 0; i < N_CLUSTERS; ++i) {
        double min_dist_sq = DBL_MAX;
        for (int j = 0; j < N_CLUSTERS; ++j) {
            double dist_sq = 0.0;
            for (int f = 0; f < n_features; ++f) {
                float diff = cuda_centroids[i * n_features + f] - cpu_centroids[j * n_features + f];
                dist_sq += diff * diff;
            }
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
        total_mse += min_dist_sq;
    }
    return (float)(total_mse / N_CLUSTERS);
}

float calculate_sparsity(const std::vector<float>& data) {
    long long zero_count = 0;
    for (float val : data) {
        if (std::abs(val) < 1e-6) zero_count++;
    }
    return (float)zero_count / data.size();
}

float run_dense_kmeans(const std::vector<float>& h_data_aos, int n_samples, int n_features, std::vector<float>& final_centroids) {
    printf("   -> Mode: [Dense SoA] (Sparsity <= %.0f%%)\n", SPARSITY_THRESHOLD * 100);

    std::vector<float> h_data_soa(n_samples * n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            h_data_soa[j * n_samples + i] = h_data_aos[i * n_features + j];
        }
    }
    std::vector<float> h_centroids(N_CLUSTERS * n_features);
    for(int i=0; i<N_CLUSTERS * n_features; ++i) h_centroids[i] = h_data_aos[i];

    float *d_data_soa, *d_centroids;
    double *d_new_centroids_double; 
    int *d_labels, *d_cluster_counts;
    
    size_t centroid_size_float = N_CLUSTERS * n_features * sizeof(float);
    size_t centroid_size_double = N_CLUSTERS * n_features * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_data_soa, h_data_soa.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, centroid_size_float));
    CUDA_CHECK(cudaMalloc(&d_new_centroids_double, centroid_size_double));
    CUDA_CHECK(cudaMalloc(&d_labels, n_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_counts, N_CLUSTERS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data_soa, h_data_soa.data(), h_data_soa.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), centroid_size_float, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int blocks = (n_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem = (N_CLUSTERS * n_features * sizeof(float)) + (N_CLUSTERS * sizeof(int));

    cudaEventRecord(start);
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_centroids, d_centroids, centroid_size_float));
        
        assignClusterKernel_SoA<<<blocks, THREADS_PER_BLOCK>>>(d_data_soa, d_labels, n_samples, n_features, N_CLUSTERS);
        
        CUDA_CHECK(cudaMemset(d_new_centroids_double, 0, centroid_size_double));
        CUDA_CHECK(cudaMemset(d_cluster_counts, 0, N_CLUSTERS * sizeof(int)));
        
        computeNewCentroidsKernel_Shared<<<blocks, THREADS_PER_BLOCK, shared_mem>>>(
            d_data_soa, d_labels, d_new_centroids_double, d_cluster_counts, n_samples, n_features, N_CLUSTERS
        );
        
        averageCentroidsKernel<<<1, THREADS_PER_BLOCK>>>(d_centroids, d_new_centroids_double, d_cluster_counts, n_features, N_CLUSTERS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    final_centroids.resize(N_CLUSTERS * n_features);
    CUDA_CHECK(cudaMemcpy(final_centroids.data(), d_centroids, final_centroids.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_data_soa); cudaFree(d_centroids); cudaFree(d_new_centroids_double);
    cudaFree(d_labels); cudaFree(d_cluster_counts);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms;
}

float run_csr_kmeans(const std::vector<float>& h_data_aos, int n_samples, int n_features, std::vector<float>& final_centroids) {
    printf("   -> Mode: [Sparse CSR] (Sparsity > %.0f%%)\n", SPARSITY_THRESHOLD * 100);

    auto csr_conv_start = std::chrono::high_resolution_clock::now();
    std::vector<float> h_values;
    std::vector<int> h_col_ind;
    std::vector<int> h_row_ptr;
    std::vector<float> h_norms; 
    
    h_row_ptr.push_back(0);
    for(int i=0; i<n_samples; ++i) {
        float row_norm = 0.0f;
        for(int j=0; j<n_features; ++j) {
            float val = h_data_aos[i * n_features + j];
            if (std::abs(val) > 1e-6) {
                h_values.push_back(val);
                h_col_ind.push_back(j);
                row_norm += val * val; 
            }
        }
        h_row_ptr.push_back(h_values.size());
        h_norms.push_back(row_norm);
    }
    auto csr_conv_end = std::chrono::high_resolution_clock::now();

    double csr_conv_time_ms =
        std::chrono::duration<double, std::milli>(csr_conv_end - csr_conv_start).count();

    printf("   -> CSR Conversion Time (CPU): %.4f ms\n", csr_conv_time_ms);

    std::vector<float> h_centroids(N_CLUSTERS * n_features);
    for(int i=0; i<N_CLUSTERS * n_features; ++i) h_centroids[i] = h_data_aos[i];

    float *d_values, *d_data_norms, *d_centroids_global_float;
    double *d_new_centroids_double, *d_centroids_global_double; 
    int *d_col_ind, *d_row_ptr;
    int *d_labels, *d_cluster_counts;
    
    size_t centroid_size_float = N_CLUSTERS * n_features * sizeof(float);
    size_t centroid_size_double = N_CLUSTERS * n_features * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_values, h_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, h_col_ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_data_norms, h_norms.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, n_samples * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_centroids_double, centroid_size_double));
    CUDA_CHECK(cudaMalloc(&d_centroids_global_double, centroid_size_double)); 
    CUDA_CHECK(cudaMalloc(&d_centroids_global_float, centroid_size_float)); 
    CUDA_CHECK(cudaMalloc(&d_cluster_counts, N_CLUSTERS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, h_col_ind.data(), h_col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_norms, h_norms.data(), h_norms.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_global_float, h_centroids.data(), centroid_size_float, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int blocks = (n_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        std::vector<float> h_curr_centroids(N_CLUSTERS * n_features);
        std::vector<float> h_curr_norms(N_CLUSTERS, 0.0f);
        
        if (iter > 0) {
            averageCentroidsKernel<<<1, THREADS_PER_BLOCK>>>(d_centroids_global_float, d_centroids_global_double, d_cluster_counts, n_features, N_CLUSTERS);
            CUDA_CHECK(cudaDeviceSynchronize()); 
        }

        CUDA_CHECK(cudaMemcpy(h_curr_centroids.data(), d_centroids_global_float, centroid_size_float, cudaMemcpyDeviceToHost));
        
        for(int k=0; k<N_CLUSTERS; ++k) {
            for(int f=0; f<n_features; ++f) {
                float val = h_curr_centroids[k*n_features + f];
                h_curr_norms[k] += val * val;
            }
        }
        
        CUDA_CHECK(cudaMemcpyToSymbol(c_centroids, h_curr_centroids.data(), centroid_size_float));
        CUDA_CHECK(cudaMemcpyToSymbol(c_centroid_norms, h_curr_norms.data(), N_CLUSTERS * sizeof(float)));

        assignClusterKernel_CSR<<<blocks, THREADS_PER_BLOCK>>>(d_values, d_col_ind, d_row_ptr, d_data_norms, d_labels, n_samples);
        
        CUDA_CHECK(cudaMemset(d_new_centroids_double, 0, centroid_size_double));
        CUDA_CHECK(cudaMemset(d_cluster_counts, 0, N_CLUSTERS * sizeof(int)));

        computeNewCentroidsKernel_CSR<<<blocks, THREADS_PER_BLOCK>>>(
            d_values, d_col_ind, d_row_ptr, d_labels, d_new_centroids_double, d_cluster_counts, n_samples
        );

        CUDA_CHECK(cudaMemcpy(d_centroids_global_double, d_new_centroids_double, centroid_size_double, cudaMemcpyDeviceToDevice));
        
    }
    
    averageCentroidsKernel<<<1, THREADS_PER_BLOCK>>>(d_centroids_global_float, d_centroids_global_double, d_cluster_counts, n_features, N_CLUSTERS);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    final_centroids.resize(N_CLUSTERS * n_features);
    CUDA_CHECK(cudaMemcpy(final_centroids.data(), d_centroids_global_float, final_centroids.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_values); cudaFree(d_col_ind); cudaFree(d_row_ptr); cudaFree(d_data_norms);
    cudaFree(d_labels); cudaFree(d_new_centroids_double); cudaFree(d_centroids_global_double); cudaFree(d_centroids_global_float); cudaFree(d_cluster_counts);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms;
}

int main() {
    printf("=== [Batch Adaptive CUDA K-Means (Verified)] ===\n");
    
    std::string data_dir = "dataset";
    std::string result_dir = "cpu_results";
    float total_time_ms = 0.0f;
    float total_mse = 0.0f;
    int success_count = 0;
    int i = 0;

    while (true) {
        std::string filename = data_dir + "/data_" + std::to_string(i) + ".bin";
        std::vector<float> h_data_raw;
        int n_samples = 0;
        int n_features = MAX_FEATURES;

        if (!load_bin(filename, h_data_raw, n_samples, n_features)) {
            if (i == 0) printf("Error: No data files found.\n");
            else printf("\n--- Finished. Processed %d files. ---\n", i);
            break;
        }

        printf("\n------------------------------------------------\n");
        printf("Processing File [%d]: %s\n", i, filename.c_str());

        auto sparsity_start = std::chrono::high_resolution_clock::now();
        float sparsity = calculate_sparsity(h_data_raw);        
        auto sparsity_end = std::chrono::high_resolution_clock::now();
        double sparsity_time_ms =
            std::chrono::duration<double, std::milli>(sparsity_end - sparsity_start).count();

        printf("   -> Sparsity: %.2f%% (Time: %.4f ms)\n",
            sparsity * 100.0f, sparsity_time_ms);

        float ms = 0.0f;
        std::vector<float> cuda_centroids;
        if (sparsity > SPARSITY_THRESHOLD) {
            ms = run_csr_kmeans(h_data_raw, n_samples, n_features, cuda_centroids);
        } else {
            ms = run_dense_kmeans(h_data_raw, n_samples, n_features, cuda_centroids);
        }

        printf("   -> Execution Time: %.4f ms\n", ms);

        std::string cpu_file = result_dir + "/centroids_" + std::to_string(i) + ".txt";
        std::vector<float> cpu_centroids;
        
        if (read_cpu_centroids(cpu_file, cpu_centroids, n_features)) {
            float mse = calculate_centroid_mse(cuda_centroids, cpu_centroids, n_features);
            printf("   -> Accuracy Check (MSE): %.6f ", mse);
            if (mse < 1.0f) printf("[PASS] \n"); 
            else printf("[WARNING: High Error] \n");
            total_mse += mse;
        } else {
            printf("   -> Accuracy Check: Skipped (CPU result not found: %s)\n", cpu_file.c_str());
        }
        
        total_time_ms += ms;
        success_count++;
        i++;
    }

    if (success_count > 0) {
        printf("\n================================================\n");
        printf("Benchmark Summary\n");
        printf("Total Files: %d\n", success_count);
        printf("Avg Execution Time: %.4f ms\n", total_time_ms / success_count);
        printf("Avg MSE Error:      %.6f\n", total_mse / success_count);
        printf("================================================\n");
    }

    return 0;
}
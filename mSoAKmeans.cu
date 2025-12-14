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

__constant__ float c_centroids[N_CLUSTERS * MAX_FEATURES];

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

bool load_bin(const std::string& filename, std::vector<float>& data, int& n_samples, int n_features) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size == 0 || size % sizeof(float) != 0) return false;
    size_t total_elements = size / sizeof(float);
    n_samples = total_elements / n_features;
    data.resize(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), size);
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
    return centroids.size() == (size_t)(N_CLUSTERS * n_features);
}

float calculate_mse(const std::vector<float>& cuda_res, const std::vector<float>& cpu_res, int n_features) {
    if (cpu_res.empty()) return -1.0f;
    double total_mse = 0.0;
    for (int i = 0; i < N_CLUSTERS; ++i) {
        double min_dist = DBL_MAX;
        for (int j = 0; j < N_CLUSTERS; ++j) {
            double dist = 0.0;
            for (int f = 0; f < n_features; ++f) {
                float diff = cuda_res[i * n_features + f] - cpu_res[j * n_features + f];
                dist += diff * diff;
            }
            if (dist < min_dist) min_dist = dist;
        }
        total_mse += min_dist;
    }
    return (float)(total_mse / N_CLUSTERS);
}

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
    double* d_new_centroids_double,
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
            atomicAdd(&d_new_centroids_double[i], (double)sum);
    }
}

__global__ void averageCentroidsKernel(float* d_centroids, double* d_new_centroids_double, int* d_cluster_counts, int n_features, int n_clusters) {
    int cid = threadIdx.x;
    if (cid < n_clusters) {
        int count = d_cluster_counts[cid];
        if (count > 0) {
            for (int f = 0; f < n_features; ++f) {
                d_centroids[cid * n_features + f] =
                    (float)(d_new_centroids_double[cid * n_features + f] / count);
            }
        }
    }
}

int main() {
    printf("=== [Optimized CUDA K-Means (Verified: SoA+Constant+Shared+Double)] ===\n");

    std::string data_dir = "dataset";
    std::string res_dir = "cpu_results";
    float total_time_ms = 0.0f;
    float total_mse = 0.0f;
    int success_count = 0;
    int i = 0;

    while (true) {
        std::string filename = data_dir + "/data_" + std::to_string(i) + ".bin";
        std::vector<float> h_data_aos;
        int n_samples = 0;
        int n_features = MAX_FEATURES;

        if (!load_bin(filename, h_data_aos, n_samples, n_features)) {
            if (i == 0) printf("Error: No data files found starting with 'data_0.bin'\n");
            else printf("\n--- Finished. Processed %d files. ---\n", success_count);
            break;
        }

        printf("\n[%d] Processing %s (Samples: %d)\n", i, filename.c_str(), n_samples);

        std::vector<float> h_data_soa(n_samples * n_features);
        for (int s = 0; s < n_samples; ++s) {
            for (int f = 0; f < n_features; ++f) {
                h_data_soa[f * n_samples + s] = h_data_aos[s * n_features + f];
            }
        }

        std::vector<float> h_centroids(N_CLUSTERS * n_features);
        for (int k = 0; k < N_CLUSTERS * n_features; ++k) h_centroids[k] = h_data_aos[k];

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
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int blocks = (n_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        size_t shared_mem_size = (N_CLUSTERS * n_features * sizeof(float)) + (N_CLUSTERS * sizeof(int));

        cudaEventRecord(start);

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            CUDA_CHECK(cudaMemcpyToSymbol(c_centroids, d_centroids, centroid_size_float));
            assignClusterKernel_SoA<<<blocks, THREADS_PER_BLOCK>>>(
                d_data_soa, d_labels, n_samples, n_features, N_CLUSTERS
            );

            CUDA_CHECK(cudaMemset(d_new_centroids_double, 0, centroid_size_double));
            CUDA_CHECK(cudaMemset(d_cluster_counts, 0, N_CLUSTERS * sizeof(int)));

            computeNewCentroidsKernel_Shared<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                d_data_soa, d_labels, d_new_centroids_double, d_cluster_counts,
                n_samples, n_features, N_CLUSTERS
            );

            averageCentroidsKernel<<<1, THREADS_PER_BLOCK>>>(
                d_centroids, d_new_centroids_double, d_cluster_counts,
                n_features, N_CLUSTERS
            );

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("   -> Execution Time: %.4f ms\n", ms);

        std::vector<float> final_centroids(N_CLUSTERS * n_features);
        CUDA_CHECK(cudaMemcpy(final_centroids.data(), d_centroids, centroid_size_float, cudaMemcpyDeviceToHost));

        std::string cpu_file = res_dir + "/centroids_" + std::to_string(i) + ".txt";
        std::vector<float> cpu_centroids;

        if (read_cpu_centroids(cpu_file, cpu_centroids, n_features)) {
            float mse = calculate_mse(final_centroids, cpu_centroids, n_features);
            printf("   -> Accuracy (MSE): %.6f ", mse);
            if (mse < 0.1f) printf("[PASS]\n");
            else printf("[WARNING: Different Convergence]\n");
            total_mse += mse;
        } else {
            printf("   -> Accuracy: Skipped (No CPU result file)\n");
        }

        total_time_ms += ms;
        success_count++;

        cudaFree(d_data_soa);
        cudaFree(d_centroids);
        cudaFree(d_new_centroids_double);
        cudaFree(d_labels);
        cudaFree(d_cluster_counts);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        i++;
    }

    if (success_count > 0) {
        printf("\n================================================\n");
        printf("Batch Processing Summary.\n");
        printf("Total Files: %d\n", success_count);
        printf("Average Execution Time: %.4f ms\n", total_time_ms / success_count);
        printf("Average Accuracy (MSE): %.6f\n", total_mse / success_count);
        printf("================================================\n");
    }

    return 0;
}

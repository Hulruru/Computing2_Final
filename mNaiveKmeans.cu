#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define N_CLUSTERS 10
#define MAX_FEATURES 100
#define MAX_ITER 10
#define THREADS_PER_BLOCK 256
#define TOLERANCE 1e-20

// CUDA API 에러 체크
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// ==========================================
// CUDA Kernels
// ==========================================

/**
 * @brief [Assignment Step] 각 데이터 포인트를 가장 가까운 중심점에 할당합니다.
 */
__global__ void assignClusterKernel_Naive(const float* d_data, const float* d_centroids, int* d_labels, int n_samples, int n_features) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n_samples) {
        float min_dist = FLT_MAX;
        int best_cluster = 0;
        
        for (int c = 0; c < N_CLUSTERS; ++c) {
            float dist = 0.0f;
            for (int f = 0; f < n_features; ++f) {
                float val = d_data[gid * n_features + f];
                float c_val = d_centroids[c * n_features + f];
                float diff = val - c_val;
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        d_labels[gid] = best_cluster;
    }
}

/**
 * 각 클러스터에 속한 샘플들의 좌표 합과 개수를 누적
 */
__global__ void computeNewCentroidsKernel_Naive(
    const float* d_data, 
    const int* d_labels, 
    double* d_new_centroids,
    int* d_cluster_counts, 
    int n_samples, int n_features) 
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n_samples) {
        int cluster_id = d_labels[gid];
        
        atomicAdd(&d_cluster_counts[cluster_id], 1);
        
        for (int f = 0; f < n_features; ++f) {
            float val = d_data[gid * n_features + f];
            atomicAdd(&d_new_centroids[cluster_id * n_features + f], (double)val);
        }
    }
}

/**
 * 누적된 합을 개수로 나누어 최종 중심점을 계산.
 */
__global__ void averageCentroidsKernel(float* d_centroids, double* d_new_centroids, int* d_cluster_counts, int n_features) {
    int cid = threadIdx.x; 
    if (cid < N_CLUSTERS) {
        int count = d_cluster_counts[cid];
        if (count > 0) {
            for (int f = 0; f < n_features; ++f) {
                d_centroids[cid * n_features + f] = (float)(d_new_centroids[cid * n_features + f] / count);
            }
        }
    }
}

// ==========================================
// Helper Functions
// ==========================================

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

bool check_convergence(const std::vector<float>& old_c, const std::vector<float>& new_c, float tolerance) {
    float total_diff = 0.0f;
    for (size_t i = 0; i < old_c.size(); ++i) {
        total_diff += std::abs(old_c[i] - new_c[i]);
    }
    return total_diff < tolerance;
}

// ==========================================
// Main Function
// ==========================================
int main() {
    printf("=== [Naive CUDA K-Means (High Precision Baseline + Early Stopping)] ===\n");
    
    std::string data_dir = "dataset";
    std::string res_dir = "cpu_results";
    float total_time_ms = 0.0f;
    int success_count = 0;
    int i = 0;

    // 데이터 파일 순차 처리 루프
    while (true) {
        std::string filename = data_dir + "/data_" + std::to_string(i) + ".bin";
        std::vector<float> h_data;
        int n_samples = 0;
        int n_features = MAX_FEATURES;

        // 1. 데이터 로드
        if (!load_bin(filename, h_data, n_samples, n_features)) {
            if (i == 0) printf("Error: No data found.\n");
            else printf("--- Finished. Processed %d files. ---\n", success_count);
            break;
        }
        printf("\n[%d] Processing %s (Samples: %d)\n", i, filename.c_str(), n_samples);

        // 초기 중심점 설정 (데이터의 앞부분 사용)
        std::vector<float> h_centroids(N_CLUSTERS * n_features);
        for(int k=0; k<N_CLUSTERS * n_features; ++k) h_centroids[k] = h_data[k];

        // 2. GPU 메모리 할당 및 데이터 복사
        float *d_data, *d_centroids;
        double *d_new_centroids;
        int *d_labels, *d_cluster_counts;
        
        size_t data_size = h_data.size() * sizeof(float);
        size_t centroid_size_float = N_CLUSTERS * n_features * sizeof(float);
        size_t centroid_size_double = N_CLUSTERS * n_features * sizeof(double);

        CUDA_CHECK(cudaMalloc(&d_data, data_size));
        CUDA_CHECK(cudaMalloc(&d_centroids, centroid_size_float));
        CUDA_CHECK(cudaMalloc(&d_new_centroids, centroid_size_double));
        CUDA_CHECK(cudaMalloc(&d_labels, n_samples * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cluster_counts, N_CLUSTERS * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), centroid_size_float, cudaMemcpyHostToDevice));

        int blocks = (n_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        std::vector<float> h_prev_centroids = h_centroids; 

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            assignClusterKernel_Naive<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_centroids, d_labels, n_samples, n_features);
            
            CUDA_CHECK(cudaMemset(d_new_centroids, 0, centroid_size_double));
            CUDA_CHECK(cudaMemset(d_cluster_counts, 0, N_CLUSTERS * sizeof(int)));
            
            computeNewCentroidsKernel_Naive<<<blocks, THREADS_PER_BLOCK>>>(d_data, d_labels, d_new_centroids, d_cluster_counts, n_samples, n_features);
            
            averageCentroidsKernel<<<1, N_CLUSTERS>>>(d_centroids, d_new_centroids, d_cluster_counts, n_features);

            // 1. 현재 계산된 중심점을 CPU로 가져옴
            std::vector<float> h_curr_centroids(N_CLUSTERS * n_features);
            CUDA_CHECK(cudaMemcpy(h_curr_centroids.data(), d_centroids, centroid_size_float, cudaMemcpyDeviceToHost));

            // 2. 이전 값과 비교
            if (check_convergence(h_prev_centroids, h_curr_centroids, TOLERANCE)) {
                break; 
            }
            h_prev_centroids = h_curr_centroids;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("    -> Execution Time: %.4f ms\n", ms);

        // 4. 결과 검증 (CPU Baseline과 비교)
        std::vector<float> final_centroids(N_CLUSTERS * n_features);
        CUDA_CHECK(cudaMemcpy(final_centroids.data(), d_centroids, centroid_size_float, cudaMemcpyDeviceToHost));

        std::string cpu_file = res_dir + "/centroids_" + std::to_string(i) + ".txt";
        std::vector<float> cpu_centroids;
        if (read_cpu_centroids(cpu_file, cpu_centroids, n_features)) {
            float mse = calculate_mse(final_centroids, cpu_centroids, n_features);
            printf("    -> Accuracy (MSE): %.6f ", mse);
            if (mse < 1.0f) printf("[PASS]\n");
            else printf("[WARNING: Different Convergence]\n");
        } else {
            printf("    -> Accuracy: Skipped (No CPU result file)\n");
        }

        total_time_ms += ms;
        success_count++;

        // 메모리 해제
        cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_new_centroids);
        cudaFree(d_labels); cudaFree(d_cluster_counts);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        i++;
    }

    if (success_count > 0) {
        printf("\n================================================\n");
        printf("Batch Processing Summary.\n");
        printf("Total Files Processed: %d\n", success_count);
        printf("Average Execution Time: %.4f ms\n", total_time_ms / success_count);
        printf("================================================\n");
    }

    return 0;
}
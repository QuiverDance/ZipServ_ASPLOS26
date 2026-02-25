/***************************************************************************
 * Copyright 2025 The ZipServ Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <iomanip>
#include <algorithm> 
#include <fstream>
#include <random>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <map>
#include <string>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
// Performance Benchmark
#define WARM_UP_ITERATION 100
#define BENCHMARK_ITERATION 2000


void SavePerformanceData(const char* filename, 
                        const char* model_name, const char* layer_name,
                        int M, int K, int N, int SplitK, 
                        float duration_cublas_tc, float tflops_cublas_tc,
                        float duration_CompGEMM, float tflops_CompGEMM,
                        float duration_notc, float tflops_notc) 
{
    FILE* fp;
    // Check if file exists
    fp = fopen(filename, "r");
    bool fileExists = (fp != NULL);
    if (fp) fclose(fp);
    
    // Open file in append mode
    fp = fopen(filename, "a");
    if (!fp) {
        printf("Error opening file for writing!\n");
        return;
    }

    // Write header if new file
    if (!fileExists) {
        fprintf(fp, "Model,Layer,M,K,N,SplitK,Kernel,Duration(ms),TFLOPS\n");
    }

    // Write data for three kernels
    const char* kernels[] = {"cuBLAS", "cuBLAS_TC", "CompGEMM"};
    const float durations[] = {duration_notc, duration_cublas_tc, duration_CompGEMM};
    const float tflops[] = {tflops_notc, tflops_cublas_tc, tflops_CompGEMM};

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%s,%s,%d,%d,%d,%d,%s,%.6f,%.6f\n", 
                model_name, layer_name,
                M, K, N, SplitK, 
                kernels[i], durations[i], tflops[i]);
    }

    fclose(fp);
}
void checkCublasError(cublasStatus_t status, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", line, status);
        exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/ms: %5.3f \t Performance/TFLOPs: %4.2f \t TotalError: %.2lf\n",
           KernelName,
           milliseconds,
           tflops,
           error);
}


void init_host_matrices_bf16(__nv_bfloat16* A_h, __nv_bfloat16* B_h, int M, int K, int N,
                           const int* custom_exponents = nullptr, unsigned seed = 12345) {
    // Default high-frequency exponent values
    // int default_exponents[7] = {123, 124, 125, 126, 127, 128, 129};
    // int default_exponents[7] = {123, 124, 125, 126, 127, 128, 130};
    int default_exponents[7] = {116, 117, 118, 119, 121, 120, 122};


    
    // Use default values if no custom exponents provided
    const int* target_exponents = custom_exponents ? custom_exponents : default_exponents;
    
    // Set decreasing probability distribution for exponents
    // Weights are {7, 6, 5, 4, 3, 2, 1}, sum = 28
    double weights[7] = {8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double total_weight = 30.0;  // 7+6+5+4+3+2+1
    
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_mantissa(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_sign(0, 1);
    std::uniform_real_distribution<double> dist_weighted(0.0, total_weight);
    
    // Exponent distribution probability: 97% use target exponents, 3% use random exponents
    std::uniform_real_distribution<float> dist_exp_choice(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_random_exp(110, 121);  // General BF16 exponent range
    
    // Initialize A - generate matrix with specific exponent distribution
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            uint8_t sign = dist_sign(gen);
            // uint8_t sign = 1;
            uint8_t mantissa = (uint8_t)(dist_mantissa(gen) * 127);
            uint8_t exponent;
            
            // 97% probability to use high-frequency exponents
            if (dist_exp_choice(gen) < 0.95f) {
                // Use weighted random selection
                double rand_val = dist_weighted(gen);
                double cumulative = 0.0;
                int idx = 0;
                
                // Find index corresponding to weight interval
                for (int w = 0; w < 7; w++) {
                    cumulative += weights[w];
                    if (rand_val < cumulative) {
                        idx = w;
                        break;
                    }
                }
                
                exponent = target_exponents[idx];
            } else {
                exponent = dist_random_exp(gen);
            }
            
            // Assemble into BF16 value
            uint16_t bf16_bits = ((sign & 0x1) << 15) | ((exponent & 0xFF) << 7) | (mantissa & 0x7F);
            A_h[i * K + j] = __ushort_as_bfloat16(bf16_bits);
        }
    }
    
    // Initialize B - use fixed value
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            // B_h[j * K + i] = __float2bfloat16(1.0);
            B_h[j * K + i] = __float2bfloat16(dis(gen));
        }
    }
}


// Calculate total error of BF16 matrix
double ComputeTotalError_BF16(const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N) {
    double total_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float a_val = __bfloat162float(A[i]);
        float b_val = __bfloat162float(B[i]);
        total_error += std::abs(a_val - b_val);
    }
    return total_error;
}



// Analyze exponent distribution of BF16 matrix
void analyzeExponentDistribution_BF16(__nv_bfloat16* matrix, int M, int K, int* top_exponents, int top_n = 7) {
    // Count exponent distribution using fixed-size array (O(1) access vs std::map O(log n))
    int exponent_counts_arr[256] = {0};

    for (int i = 0; i < M * K; i++) {
        uint16_t bits = __bfloat16_as_ushort(matrix[i]);
        uint8_t exponent = (bits >> 7) & 0xFF;
        exponent_counts_arr[exponent]++;
    }

    // Build frequency pairs (ascending key order, same as std::map iteration) and sort
    std::vector<std::pair<int, int>> exponent_counts;
    exponent_counts.reserve(256);
    for (int e = 0; e < 256; ++e) {
        if (exponent_counts_arr[e] > 0) {
            exponent_counts.push_back(std::make_pair(e, exponent_counts_arr[e]));
        }
    }
    std::sort(exponent_counts.begin(), exponent_counts.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.second > b.second;  // Sort by frequency in descending order
              });

    // Extract original values of top_n exponents
    std::vector<int> original_top;
    for (int i = 0; i < std::min(top_n, (int)exponent_counts.size()); i++) {
        original_top.push_back(exponent_counts[i].first);
    }

    // Fill with default values if fewer than top_n exponents found
    while (original_top.size() < top_n) {
        original_top.push_back(127 - original_top.size());
    }

    // Sort original top exponents
    std::sort(original_top.begin(), original_top.end());

    // Check continuity and handle outliers
    bool has_outlier = false;
    std::vector<int> continuous_top;
    int start_exp = original_top[0];

    // Attempt to construct continuous exponent range
    for (int try_offset = 0; try_offset <= original_top.back() - start_exp; try_offset++) {
        continuous_top.clear();
        int current_exp = start_exp + try_offset;
        int found_count = 0;

        // Attempt to construct continuous sequence starting with current_exp
        for (int i = 0; i < top_n; i++) {
            if (current_exp + i < 256 && exponent_counts_arr[current_exp + i] > 0) {
                continuous_top.push_back(current_exp + i);
                found_count++;
            }
        }

        // If enough continuous exponents found
        if (found_count >= top_n) {
            break;
        }
    }
    
    // If not enough continuous exponents found, use most frequent continuous interval
    if (continuous_top.size() < top_n) {
        continuous_top.clear();
        int best_start = original_top[0];
        int max_length = 1;
        int current_length = 1;
        
        // Find longest continuous interval
        for (size_t i = 1; i < original_top.size(); i++) {
            if (original_top[i] == original_top[i-1] + 1) {
                current_length++;
                if (current_length > max_length) {
                    max_length = current_length;
                    best_start = original_top[i] - current_length + 1;
                }
            } else {
                current_length = 1;
            }
        }
        
        // Construct result
        for (int i = 0; i < top_n; i++) {
            if (i < max_length) {
                continuous_top.push_back(best_start + i);
            } else {
                // Fill remaining positions
                continuous_top.push_back(continuous_top.back() + 1);
            }
        }
        
        has_outlier = true;
    }
    
    // Check if adjustment needed to get fully continuous sequence
    if (continuous_top.size() > top_n) {
        continuous_top.resize(top_n);
    }
    
    // Copy result to output array
    for (int i = 0; i < top_n; i++) {
        top_exponents[i] = continuous_top[i];
    }
    
    // // Output warning information
    // if (has_outlier) {
    //     std::cerr << "WARNING: Original top exponents were not continuous. Constructed continuous range: ";
    //     for (int i = 0; i < top_n; i++ ) {
    //         std::cerr << top_exponents[i] << " ";
    //     }
    //     std::cerr << std::endl;
        
    //     std::cerr << "Original top exponents (sorted): ";
    //     for (int exp : original_top) {
    //         std::cerr << exp << " ";
    //     }
    //     std::cerr << std::endl;
    // }
}



int CompressBF16MatrixTripleBitmap_Host(
    __nv_bfloat16* A_bf16,
    int M, int K,
    int tile_M, int tile_M_median, int tile_M_global,
    int tile_K, int tile_K_median, int tile_K_global,
    const int* top_exponents,
    __nv_bfloat16** compressed_full,
    uint8_t** sign_mantissa,
    uint64_t** bitmap1, uint64_t** bitmap2, uint64_t** bitmap3,
    int** TileOffsets, int** TileOffsets_median, int** TileOffsets_global,
    int& max_high_freq_count, int& max_full_count)
{
    // Similar to original C++ implementation, but directly handles BF16 data
    // This implements the BF16 version corresponding to CompressBF16MatrixByTripleBitmap
    
    // Calculate tile counts at all levels
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_median_tiles_M = M / tile_M_median;
    int num_median_tiles_K = K / tile_K_median;
    int num_median_tiles = num_median_tiles_M * num_median_tiles_K;

    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // Allocate memory
    *compressed_full = (__nv_bfloat16*)malloc(M * K * sizeof(__nv_bfloat16) + num_global_tiles * 8 * sizeof(__nv_bfloat16));
    *sign_mantissa = (uint8_t*)malloc(M * K + num_global_tiles * 16);
    *bitmap1 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    *bitmap2 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    *bitmap3 = (uint64_t*)malloc(num_tiles * sizeof(uint64_t));
    *TileOffsets = (int*)malloc(num_tiles * 2 * sizeof(int));
    *TileOffsets_median = (int*)malloc(num_median_tiles * 2 * sizeof(int));
    *TileOffsets_global = (int*)malloc((num_global_tiles + 1) * 2 * sizeof(int));

    if (*compressed_full == nullptr || *sign_mantissa == nullptr ||
        *bitmap1 == nullptr || *bitmap2 == nullptr || *bitmap3 == nullptr ||
        *TileOffsets == nullptr || *TileOffsets_median == nullptr || 
        *TileOffsets_global == nullptr) {
        return -1;
    }

    // Initialize memory
    memset(*compressed_full, 0, M * K * sizeof(__nv_bfloat16) + num_global_tiles * 8 * sizeof(__nv_bfloat16));
    memset(*sign_mantissa, 0, M * K + num_global_tiles * 16);
    memset(*bitmap1, 0, num_tiles * sizeof(uint64_t));
    memset(*bitmap2, 0, num_tiles * sizeof(uint64_t));
    memset(*bitmap3, 0, num_tiles * sizeof(uint64_t));
    memset(*TileOffsets, 0, num_tiles * 2 * sizeof(int));
    memset(*TileOffsets_median, 0, num_median_tiles * 2 * sizeof(int));
    memset(*TileOffsets_global, 0, (num_global_tiles + 1) * 2 * sizeof(int));

    // Current offset positions
    int full_offset = 0;
    int sign_mantissa_offset = 0;
    int tile_idx = 0;
    int median_offset_idx = 0;
    std::vector<int> global_high_freq_counts(num_global_tiles + 1, 0);
    std::vector<int> global_full_counts(num_global_tiles + 1, 0);

    max_high_freq_count = 0;
    max_full_count = 0;

    // Compression statistics
    int high_freq_total = 0;
    int full_value_total = 0;
    
    // Padding counts
    int total_high_freq_padding = 0;
    int total_full_padding = 0;
    
    // Compression process - similar to original C++ implementation, but directly operates on BF16 data
    // Implements compression logic similar to CompressBF16MatrixByTripleBitmap...
    
    // Iterate over all global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;
            int global_high_freq_count = 0;
            int global_full_count = 0;
            
            int median_high_freq_count = 0;
            int median_full_count = 0;
            
            // Store starting value of medium tile offsets
            (*TileOffsets_median)[median_offset_idx * 2] = 0;
            (*TileOffsets_median)[median_offset_idx * 2 + 1] = 0;
            median_offset_idx++;
            
            // Iterate over medium tiles
            for (int median_tile_m = 0; median_tile_m < tile_M_global / tile_M_median; ++median_tile_m) {
                for (int median_tile_k = 0; median_tile_k < tile_K_global / tile_K_median; ++median_tile_k) {
                    int median_row_start = global_row_start + median_tile_m * tile_M_median;
                    int median_col_start = global_col_start + median_tile_k * tile_K_median;
                    
                    int local_median_high_freq = 0;
                    int local_median_full = 0;
                    
                    // Process small tile groups within medium tile
                    for (int local_tile_m_group = 0; local_tile_m_group < tile_M_median / tile_M; local_tile_m_group += 2) {
                        for (int local_tile_k_group = 0; local_tile_k_group < tile_K_median / tile_K; local_tile_k_group += 2) {
                            // Process 2x2 small tile group
                            for (int j = 0; j < 2; ++j) {
                                for (int i = 0; i < 2; ++i) {
                                    int local_tile_k = local_tile_k_group + j;
                                    int local_tile_m = local_tile_m_group + i;

                                    int col_start = median_col_start + local_tile_k * tile_K;
                                    int row_start = median_row_start + local_tile_m * tile_M;

                                    uint64_t tile_bitmap1 = 0;
                                    uint64_t tile_bitmap2 = 0;
                                    uint64_t tile_bitmap3 = 0;
                                    int tile_high_freq_count = 0;
                                    int tile_full_count = 0;

                                    // Process each element in small tile
                                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                                        for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                                            int row = row_start + row_offset;
                                            int col = col_start + col_offset;
                                            int pos = row_offset * tile_K + col_offset;

                                            if (row < M && col < K) {
                                                __nv_bfloat16 val = A_bf16[row * K + col];
                                                
                                                // Extract BF16 components
                                                uint16_t bf16_bits = __bfloat16_as_ushort(val);
                                                uint8_t sign = (bf16_bits >> 15) & 0x1;
                                                uint8_t exponent = (bf16_bits >> 7) & 0xFF;
                                                uint8_t mantissa = bf16_bits & 0x7F;
                                                
                                                // Determine if exponent is in high-frequency list
                                                int exp_idx = -1;
                                                for (int e = 0; e < 7; e++) {
                                                    if (exponent == top_exponents[e]) {
                                                        exp_idx = e;
                                                        break;
                                                    }
                                                }
                                                
                                                bool is_high_freq = (exp_idx >= 0);
                                                
                                                if (is_high_freq) {
                                                    // High-frequency exponent element
                                                    int bitmap_code = exp_idx + 1;  // 1-7
                                                    
                                                    // Set three bitmaps
                                                    tile_bitmap1 |= ((bitmap_code & 0x1) ? 1ULL << pos : 0);
                                                    tile_bitmap2 |= ((bitmap_code & 0x2) ? 1ULL << pos : 0);
                                                    tile_bitmap3 |= ((bitmap_code & 0x4) ? 1ULL << pos : 0);
                                                    
                                                    // Store sign bit + mantissa
                                                    uint8_t combined = ((sign & 0x1) << 7) | (mantissa & 0x7F);
                                                    (*sign_mantissa)[sign_mantissa_offset++] = combined;
                                                    
                                                    tile_high_freq_count++;
                                                    local_median_high_freq++;
                                                    global_high_freq_count++;
                                                    high_freq_total++;
                                                } else {
                                                    // Non-high-frequency exponent element
                                                    (*compressed_full)[full_offset++] = val;
                                                    
                                                    tile_full_count++;
                                                    local_median_full++;
                                                    global_full_count++;
                                                    full_value_total++;
                                                }
                                            }
                                        }
                                    }

                                    // Store bitmaps and element counts
                                    (*bitmap1)[tile_idx] = tile_bitmap1;
                                    (*bitmap2)[tile_idx] = tile_bitmap2;
                                    (*bitmap3)[tile_idx] = tile_bitmap3;
                                    (*TileOffsets)[tile_idx * 2] = tile_high_freq_count;
                                    (*TileOffsets)[tile_idx * 2 + 1] = tile_full_count;
                                    ++tile_idx;
                                }
                            }
                        }
                    }
                    
                    // Update medium tile offsets
                    if (median_tile_m < (tile_M_global / tile_M_median - 1) || 
                        median_tile_k < (tile_K_global / tile_K_median - 1)) {
                        
                        median_high_freq_count += local_median_high_freq;
                        median_full_count += local_median_full;
                        
                        (*TileOffsets_median)[median_offset_idx * 2] = median_high_freq_count;
                        (*TileOffsets_median)[median_offset_idx * 2 + 1] = median_full_count;
                        median_offset_idx++;
                    }
                }
            }
            
            // Add padding
            int high_freq_padding = (16 - (global_high_freq_count % 16)) % 16;
            for (int p = 0; p < high_freq_padding; ++p) {
                (*sign_mantissa)[sign_mantissa_offset++] = 0;
            }
            global_high_freq_count += high_freq_padding;
            high_freq_total += high_freq_padding;
            total_high_freq_padding += high_freq_padding;
            
            int full_padding = (8 - (global_full_count % 8)) % 8;
            for (int p = 0; p < full_padding; ++p) {
                (*compressed_full)[full_offset++] = __float2bfloat16(0.0f);
            }
            global_full_count += full_padding;
            full_value_total += full_padding;
            total_full_padding += full_padding;
            
            // Record element count for this global tile
            global_high_freq_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_high_freq_count;
            global_full_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_full_count;
            
            // Update max element counts
            if (global_high_freq_count > max_high_freq_count) {
                max_high_freq_count = global_high_freq_count;
            }
            if (global_full_count > max_full_count) {
                max_full_count = global_full_count;
            }
        }
    }
    
    // Calculate global tile cumulative offsets
    (*TileOffsets_global)[0] = 0;
    (*TileOffsets_global)[1] = 0;
    
    for (int i = 1; i <= num_global_tiles; ++i) {
        global_high_freq_counts[i] += global_high_freq_counts[i - 1];
        global_full_counts[i] += global_full_counts[i - 1];
        
        (*TileOffsets_global)[i * 2] = global_high_freq_counts[i];
        (*TileOffsets_global)[i * 2 + 1] = global_full_counts[i];
    }
    
    // Resize arrays
    *sign_mantissa = (uint8_t*)realloc(*sign_mantissa, sign_mantissa_offset);
    *compressed_full = (__nv_bfloat16*)realloc(*compressed_full, full_offset * sizeof(__nv_bfloat16));
    
    // Following is detailed debugging information
    std::cout << "\n========= Detailed Compression Debug Info =========\n";
    
    // 1. Basic statistics
    std::cout << "Matrix size: " << M << "x" << K << "\n";
    std::cout << "Tiling info: \n";
    std::cout << "  Small tile size: " << tile_M << "x" << tile_K << ", count: " << num_tiles << " (" << num_tiles_M << "x" << num_tiles_K << ")\n";
    std::cout << "  Medium tile size: " << tile_M_median << "x" << tile_K_median << ", count: " << num_median_tiles << " (" << num_median_tiles_M << "x" << num_median_tiles_K << ")\n";
    std::cout << "  Global tile size: " << tile_M_global << "x" << tile_K_global << ", count: " << num_global_tiles << " (" << num_global_tiles_M << "x" << num_global_tiles_K << ")\n";
    
    // 2. High-frequency exponent list
    std::cout << "High-freq exponent list (7 values): ";
    for (int i = 0; i < 7; ++i) {
        std::cout << top_exponents[i] << " ";
    }
    std::cout << "\n";
    
    // 3. Element statistics
    std::cout << "\nElement statistics:\n";
    std::cout << "  High-frequency exponent elements: " << high_freq_total << " (" 
             << std::fixed << std::setprecision(2) 
             << (100.0f * high_freq_total / (M * K)) << "%)\n";
    std::cout << "  Non-high-frequency exponent elements: " << full_value_total << " (" 
             << std::fixed << std::setprecision(2) 
             << (100.0f * full_value_total / (M * K)) << "%)\n";
    std::cout << "  Padding count in high-frequency elements: " << total_high_freq_padding << " elements\n";
    std::cout << "  Padding count in non-high-frequency elements: " << total_full_padding << " elements\n";
    std::cout << "  Max high-frequency elements (per global tile): " << max_high_freq_count << "\n";
    std::cout << "  Max non-high-frequency elements (per global tile): " << max_full_count << "\n";
    
    // 4. Memory usage
    std::cout << "\nMemory usage:\n";
    std::cout << "  Sign+mantissa array size: " << sign_mantissa_offset << " bytes\n";
    std::cout << "  Full BF16 value array size: " << full_offset * sizeof(__nv_bfloat16) << " bytes\n";
    std::cout << "  Bitmap size: " << num_tiles * sizeof(uint64_t) * 3 << " bytes\n";
    std::cout << "  Small tile offset array size: " << num_tiles * 2 * sizeof(int) << " bytes\n";
    std::cout << "  Medium tile offset array size: " << num_median_tiles * 2 * sizeof(int) << " bytes\n";
    std::cout << "  Global tile offset array size: " << (num_global_tiles + 1) * 2 * sizeof(int) << " bytes\n";
    
    // 5. Compression ratio
    size_t original_size = M * K * sizeof(uint16_t);  // Original BF16 size
    size_t compressed_size = 
        sign_mantissa_offset + 
        (full_offset * sizeof(__nv_bfloat16)) +
        (num_tiles * sizeof(uint64_t) * 3) +  // Three bitmaps
        // (num_tiles * 2 * sizeof(int)) +   // TileOffsets
        (num_median_tiles * 2 * sizeof(int)) +  // TileOffsets_median
        ((num_global_tiles + 1) * 2 * sizeof(int));  // TileOffsets_global
    
    float compression_ratio = static_cast<float>(original_size) / compressed_size;
    
    std::cout << "  Original size: " << original_size << " bytes\n";
    std::cout << "  Compressed size: " << compressed_size << " bytes\n";
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(3) << compression_ratio << ":1\n";
    
    // 6. Print partial array contents for verification
    int print_limit = std::min(20, num_tiles); // Limit number of entries to print
    
    std::cout << "\nBitmap examples (first " << print_limit << " small tiles):\n";
    std::cout << "  TileID   Bitmap1           Bitmap2           Bitmap3           HF-Count  NHF-Count\n";
    for (int i = 0; i < print_limit; ++i) {
        std::cout << "  " << std::setw(7) << i << "   " 
                 << std::hex << std::setw(16) << std::setfill('0') << (*bitmap1)[i] << "   "
                 << std::hex << std::setw(16) << std::setfill('0') << (*bitmap2)[i] << "   "
                 << std::hex << std::setw(16) << std::setfill('0') << (*bitmap3)[i] << "   "
                 << std::dec << std::setfill(' ') << std::setw(8) << (*TileOffsets)[i*2] << "  "
                 << std::setw(10) << (*TileOffsets)[i*2+1] << "\n";
    }
    
    print_limit = std::min(10, num_median_tiles);
    std::cout << "\nMedium tile offset examples (first " << print_limit << " medium tiles):\n";
    std::cout << "  TileID   HF-Cumul-Count  NHF-Cumul-Count\n";
    for (int i = 0; i < print_limit; ++i) {
        std::cout << "  " << std::setw(7) << i << "   " 
                 << std::setw(12) << (*TileOffsets_median)[i*2] << "   "
                 << std::setw(14) << (*TileOffsets_median)[i*2+1] << "\n";
    }
    
    print_limit = std::min(5, num_global_tiles);
    std::cout << "\nGlobal tile offset examples (first " << print_limit << " global tiles):\n";
    std::cout << "  TileID   HF-Cumul-Count  NHF-Cumul-Count\n";
    for (int i = 0; i <= print_limit; ++i) { // Include initial offset (0)
        std::cout << "  " << std::setw(7) << i << "   " 
                 << std::setw(12) << (*TileOffsets_global)[i*2] << "   "
                 << std::setw(14) << (*TileOffsets_global)[i*2+1] << "\n";
    }
    
    // 7. Print partial contents of sign_mantissa and compressed_full
    print_limit = std::min(20, sign_mantissa_offset);
    std::cout << "\nSign+mantissa array examples (first " << print_limit << " bytes):\n";
    std::cout << "  Index   Hex        Sign  Mantissa\n";
    for (int i = 0; i < print_limit; ++i) {
        uint8_t value = (*sign_mantissa)[i];
        uint8_t sign = (value >> 7) & 0x1;
        uint8_t mantissa = value & 0x7F;
        std::cout << "  " << std::setw(5) << i << "   " 
                 << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value) << "       "
                 << std::dec << std::setfill(' ') << static_cast<int>(sign) << "     "
                 << static_cast<int>(mantissa) << "\n";
    }
    
    print_limit = std::min(10, full_offset);
    std::cout << "\nFull BF16 value array examples (first " << print_limit << " values):\n";
    std::cout << "  Index   BF16 bits  Sign  Exponent  Mantissa  Float\n";
    for (int i = 0; i < print_limit; ++i) {
        __nv_bfloat16 bf16_val = (*compressed_full)[i];
        uint16_t bits = __bfloat16_as_ushort(bf16_val);
        uint8_t sign = (bits >> 15) & 0x1;
        uint8_t exponent = (bits >> 7) & 0xFF;
        uint8_t mantissa = bits & 0x7F;
        float f_val = __bfloat162float(bf16_val);
        
        std::cout << "  " << std::setw(5) << i << "   " 
                 << std::hex << std::setw(4) << std::setfill('0') << bits << "       "
                 << std::dec << std::setfill(' ') << static_cast<int>(sign) << "     "
                 << std::setw(3) << static_cast<int>(exponent) << "    "
                 << std::setw(3) << static_cast<int>(mantissa) << "    "
                 << std::fixed << std::setprecision(6) << f_val << "\n";
    }
    std::cout << "======================================\n";
    return num_global_tiles;
}
// CPU version of BF16 initialization function
int InitBF16MatrixTripleBitmap_Host(
    __nv_bfloat16* A_bf16,
    int M, int K,
    int tile_M, int tile_M_median, int tile_M_global,
    int tile_K, int tile_K_median, int tile_K_global,
    __nv_bfloat16** top_exponents, 
    __nv_bfloat16** compressed_full,
    uint8_t** sign_mantissa,
    uint64_t** bitmap1, uint64_t** bitmap2, uint64_t** bitmap3,
    int** TileOffsets, int** TileOffsets_median, int** TileOffsets_global,
    int& max_high_freq_count, int& max_full_count, uint8_t& start_exp)
{
    // Analyze exponent distribution of BF16 matrix
    int top_exponent_values[7] = {0};
    analyzeExponentDistribution_BF16(A_bf16, M, K, top_exponent_values);
    start_exp = top_exponent_values[0] - 1;
    // Allocate top_exponents memory and convert to BF16
    *top_exponents = (__nv_bfloat16*)malloc(7 * sizeof(__nv_bfloat16));
    if (*top_exponents == nullptr) return -1;
    
    for (int i = 0; i < 7; i++) {
        // Create BF16 value with exponent part only
        uint16_t exp_bits = (top_exponent_values[i] & 0xFF) << 7;
        (*top_exponents)[i] = __ushort_as_bfloat16(exp_bits);
    }
    
    // Call compression function (needs reimplementation for direct BF16 operations)
    return CompressBF16MatrixTripleBitmap_Host(
        A_bf16, M, K, 
        tile_M, tile_M_median, tile_M_global,
        tile_K, tile_K_median, tile_K_global,
        top_exponent_values, compressed_full, sign_mantissa,
        bitmap1, bitmap2, bitmap3,
        TileOffsets, TileOffsets_median, TileOffsets_global,
        max_high_freq_count, max_full_count
    );
}




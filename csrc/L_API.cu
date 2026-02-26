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
#include "./MatMulUtilities.cuh"
#include "./Reduction_Kernel.cuh"
#include "./L_Kernel.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

cudaError_t LaunchKernelWithConfig_4Param(
    cudaStream_t stream,
    const uint8_t* SignMantissa, const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1, const uint64_t* Bitmap2, const uint64_t* Bitmap3,
    const int* TileOffsets_Median, const int* TileOffsets_Global,
    const int max_high_freq_count, const int max_full_count,
    const uint8_t start_exp, const __nv_bfloat16* B, __nv_bfloat16* OutputPTR,
    const int M_Global, const int N_Global, const int K_Global, int Split_K)
{
    using ConfigType = TilingConfigBF16TripleBitmap<4, 1, 1, 1>;
    
    static int SHMEM_SZ = max(
        (max_high_freq_count * sizeof(uint8_t)*2) + 
        (max_full_count * sizeof(__nv_bfloat16)*2) +
        (ConfigType::TILE_N * TILE_K * sizeof(__nv_bfloat16) * 2) + 
        (ConfigType::TILE_BITMAP_M_V3 * ConfigType::TILE_BITMAP_K_V3 * sizeof(uint64_t) * 6),
        (ConfigType::TILE_M + PADDING_SHARED_MEM_FOR_C) * ConfigType::TILE_N * sizeof(float));
    
    int dimN = (N_Global + ConfigType::TILE_N - 1) / ConfigType::TILE_N;
    int dimM = M_Global * Split_K / ConfigType::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * ConfigType::BLOCK_WARPS, 1, 1);
    
    // === Key modification: Choose Fast or Safe version based on N_Global ===
    if (N_Global % ConfigType::TILE_N2 == 0) {
        // N is a multiple of TILE_N, use Fast version
        cudaFuncSetAttribute(BF16TripleBitmap_MM_Kernel_Fast<ConfigType>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        BF16TripleBitmap_MM_Kernel_Fast<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    } else {
        // When N is not a multiple of TILE_N, use the Safe version
        cudaFuncSetAttribute(BF16TripleBitmap_MM_Kernel_Safe<ConfigType>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        BF16TripleBitmap_MM_Kernel_Safe<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    }
    
    return cudaGetLastError();
}

template<int BLOCK_COL_WARPS>
cudaError_t LaunchKernelWithConfig_3Param(
    cudaStream_t stream,
    const uint8_t* SignMantissa, const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1, const uint64_t* Bitmap2, const uint64_t* Bitmap3,
    const int* TileOffsets_Median, const int* TileOffsets_Global,
    const int max_high_freq_count, const int max_full_count,
    const uint8_t start_exp, const __nv_bfloat16* B, __nv_bfloat16* OutputPTR,
    const int M_Global, const int N_Global, const int K_Global, int Split_K)
{
    using ConfigType = TilingConfigBF16TripleBitmap<4, 1, BLOCK_COL_WARPS>;
    
    static int SHMEM_SZ = max(
        (max_high_freq_count * sizeof(uint8_t)*2) + 
        (max_full_count * sizeof(__nv_bfloat16)*2) +
        (ConfigType::TILE_N * TILE_K * sizeof(__nv_bfloat16) * 2) + 
        (ConfigType::TILE_BITMAP_M_V3 * ConfigType::TILE_BITMAP_K_V3 * sizeof(uint64_t) * 6),
        (ConfigType::TILE_M + PADDING_SHARED_MEM_FOR_C) * ConfigType::TILE_N * sizeof(float));
    
    int dimN = (N_Global + ConfigType::TILE_N - 1) / ConfigType::TILE_N;
    int dimM = M_Global * Split_K / ConfigType::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * ConfigType::BLOCK_WARPS, 1, 1);
    
    // === Key modification: Choose Fast or Safe version based on N_Global ===
    if (N_Global % ConfigType::TILE_N == 0) {
        // N is a multiple of TILE_N, use Fast version
        cudaFuncSetAttribute(BF16TripleBitmap_MM_Kernel_Fast<ConfigType>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        BF16TripleBitmap_MM_Kernel_Fast<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    } else {
        // When N is not a multiple of TILE_N, use the Safe version
        cudaFuncSetAttribute(BF16TripleBitmap_MM_Kernel_Safe<ConfigType>, 
                           cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
        BF16TripleBitmap_MM_Kernel_Safe<ConfigType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
            SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
            TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR, 
            M_Global, N_Global, K_Global, Split_K);
    }
    
    return cudaGetLastError();
}

cudaError_t BF16TripleBitmap_MM_API(
    cudaStream_t stream,
    const uint8_t* SignMantissa,          
    const __nv_bfloat16* CompressedFull,  
    const uint64_t* Bitmap1,              
    const uint64_t* Bitmap2,              
    const uint64_t* Bitmap3,              
    const int* TileOffsets_Median,        
    const int* TileOffsets_Global,        
    const int max_high_freq_count,        
    const int max_full_count,             
    const uint8_t start_exp,
    const __nv_bfloat16* B,               
    __nv_bfloat16* C,                     
    const int M_Global,                   
    const int N_Global,                   
    const int K_Global,                   
    __nv_bfloat16* Reduction_Workspace,   
    int Split_K)                          
{
    __nv_bfloat16* OutputPTR;
    if (Split_K == 1)
        OutputPTR = C;
    else
        OutputPTR = Reduction_Workspace;
    
    // === Key modification: Select different Config types based on N_Global ===
    cudaError_t error;
    
    if (N_Global <= 8) {
        // === Special case: Use 4-parameter configuration ===
        error = LaunchKernelWithConfig_4Param(stream, SignMantissa, CompressedFull,
            Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
            M_Global, N_Global, K_Global, Split_K);
    }
    else if (N_Global > 128) {
        // Greater than 128, use fixed 3-parameter configuration BLOCK_COL_WARPS=8 (TILE_N=128)
        error = LaunchKernelWithConfig_3Param<8>(stream, SignMantissa, CompressedFull,
            Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
            max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
            M_Global, N_Global, K_Global, Split_K);
    }
    else {
        // 9 <= N_Global <= 128, use 3-parameter configuration, BLOCK_COL_WARPS = (N_Global + 15) / 16
        int block_col_warps = (N_Global + 15) / 16;
        
        switch (block_col_warps) {
            case 1:
                error = LaunchKernelWithConfig_3Param<1>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 2:
                error = LaunchKernelWithConfig_3Param<2>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 3:
                error = LaunchKernelWithConfig_3Param<3>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 4:
                error = LaunchKernelWithConfig_3Param<4>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 5:
                error = LaunchKernelWithConfig_3Param<5>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 6:
                error = LaunchKernelWithConfig_3Param<6>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 7:
                error = LaunchKernelWithConfig_3Param<7>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
            case 8:
            default:
                error = LaunchKernelWithConfig_3Param<8>(stream, SignMantissa, CompressedFull,
                    Bitmap1, Bitmap2, Bitmap3, TileOffsets_Median, TileOffsets_Global,
                    max_high_freq_count, max_full_count, start_exp, B, OutputPTR,
                    M_Global, N_Global, K_Global, Split_K);
                break;
        }
    }
    
    if (error != cudaSuccess)
        return error;
    
    // If using Split-K, perform reduction
    if (Split_K > 1) {
        dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction_BF16<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    
    return cudaGetLastError();
}


// API function
cudaError_t BF16TripleBitmap_Decompress_API(
    cudaStream_t stream,
    const uint8_t* SignMantissa,
    const __nv_bfloat16* CompressedFull,
    const uint64_t* Bitmap1,
    const uint64_t* Bitmap2,
    const uint64_t* Bitmap3,
    /*const int* TileOffsets,*/
    const int* TileOffsets_Median,
    const int* TileOffsets_Global,
    /*const __nv_bfloat16* TopExponents,*/
    const int max_high_freq_count,
    const int max_full_count,
    const uint8_t start_exp,
    __nv_bfloat16* Output,
    const int M_Global,
    const int K_Global)
{
    // Validate input parameters
    if (M_Global % 64 != 0 || K_Global % 64 != 0) {
        printf("Error: Matrix dimensions must be multiples of 64. Got M=%d, K=%d\n", M_Global, K_Global);
        return cudaErrorInvalidValue;
    }
    
    // Calculate grid dimensions
    int num_global_tiles_m = M_Global / 64;
    // int num_global_tiles_m = M_Global / 128;
    // int num_global_tiles_m = M_Global / 256;


    int num_global_tiles_k = K_Global / 64;
    
    dim3 GridDim(num_global_tiles_k, num_global_tiles_m, 1);
    dim3 BlockDim(WARP_SIZE * 4, 1, 1); // 4 warps per block
    // dim3 BlockDim(WARP_SIZE * 8, 1, 1); // 4 warps per block
    // dim3 BlockDim(WARP_SIZE * 16, 1, 1); // 4 warps per block


    
    // Calculate shared memory size
    using Config = TilingConfigBF16TripleBitmap<4, 1, 4>; // Reuse existing configuration
    const int bitmap_size = Config::TILE_BITMAP_M_V3 * Config::TILE_BITMAP_K_V3;
    
    int SHMEM_SZ = 
        (bitmap_size * sizeof(uint64_t) * 3) +           // Three bitmaps
        (max_high_freq_count * sizeof(uint8_t)) +        // sign_mantissa
        (max_full_count * sizeof(__nv_bfloat16)) +       // compressed_full
        (64 * (64+PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16)) + 
        256 * sizeof(uint8_t);               // Decompression output buffer
    // int SHMEM_SZ = 
    //     (bitmap_size * sizeof(uint64_t) * 3) +           // three bitmaps
    //     (max_high_freq_count * sizeof(uint8_t)) +        // sign_mantissa
    //     (max_full_count * sizeof(__nv_bfloat16)) +       // compressed_full
    //     (128 * (64+PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16)) + 
    //     256 * sizeof(uint8_t);               // decompression output buffer    
    // int SHMEM_SZ = 
    //     (bitmap_size * sizeof(uint64_t) * 3) +           // three bitmaps
    //     (max_high_freq_count * sizeof(uint8_t)) +        // sign_mantissa
    //     (max_full_count * sizeof(__nv_bfloat16)) +       // compressed_full
    //     (256 * (64+PADDING_SHARED_MEM_FOR_DECOMP) * sizeof(__nv_bfloat16)) + 
    //     256 * sizeof(uint8_t);               // decompression output buffer    
        // Set dynamic shared memory
    cudaFuncSetAttribute(
        BF16TripleBitmap_Decompress_Kernel<Config>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    
    // printf("Launching decompress kernel with grid (%d, %d), block (%d), SHMEM=%d KB\n", 
    //        num_global_tiles_k, num_global_tiles_m, WARP_SIZE * 8, SHMEM_SZ / 1024);
    
    // Launch kernel
    BF16TripleBitmap_Decompress_Kernel<Config><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        SignMantissa, CompressedFull, Bitmap1, Bitmap2, Bitmap3,
        /*TileOffsets,*/ TileOffsets_Median, TileOffsets_Global, /*TopExponents,*/
        max_high_freq_count, max_full_count, start_exp,
        Output, M_Global, K_Global);
    
    return cudaGetLastError();
}

// CPU function for BF16 matrix compression initialization
__host__ int InitBF16MatrixTripleBitmap(
    __nv_bfloat16* A_bf16,
    int M,
    int K,
    int tile_M,  // 8
    int tile_M_median,  // 16
    int tile_M_global,  // 64
    int tile_K,  // 8
    int tile_K_median,  // 64
    int tile_K_global,  // 64
    const int* top_exponents,  // 7 top frequent exponent values
    uint8_t** sign_mantissa,   // Sign bit + mantissa for high frequency exponents
    __nv_bfloat16** compressed_full, // Complete BF16 for non-high frequency exponents
    uint64_t** bitmap1,        // First bitmap
    uint64_t** bitmap2,        // Second bitmap 
    uint64_t** bitmap3,        // Third bitmap
    int** TileOffsets,         // Small tile offsets
    int** TileOffsets_median,  // Medium tile offsets
    int** TileOffsets_global,  // Large tile offsets
    int& max_high_freq_count,  // Return max high frequency element count
    int& max_full_count)       // Return max non-high frequency element count
{
    // Calculate number of tiles
    int num_tiles_M = M / tile_M;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_M * num_tiles_K;
    
    int num_median_tiles_M = M / tile_M_median;
    int num_median_tiles_K = K / tile_K_median;
    int num_median_tiles = num_median_tiles_M * num_median_tiles_K;

    int num_global_tiles_M = M / tile_M_global;
    int num_global_tiles_K = K / tile_K_global;
    int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    // Memory allocation
    *compressed_full = (__nv_bfloat16*)malloc(M * K * sizeof(__nv_bfloat16));
    *sign_mantissa = (uint8_t*)malloc(M * K);
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
    memset(*compressed_full, 0, M * K * sizeof(__nv_bfloat16));
    memset(*sign_mantissa, 0, M * K);
    memset(*bitmap1, 0, num_tiles * sizeof(uint64_t));
    memset(*bitmap2, 0, num_tiles * sizeof(uint64_t));
    memset(*bitmap3, 0, num_tiles * sizeof(uint64_t));
    memset(*TileOffsets, 0, num_tiles * 2 * sizeof(int));
    memset(*TileOffsets_median, 0, num_median_tiles * 2 * sizeof(int));
    memset(*TileOffsets_global, 0, (num_global_tiles + 1) * 2 * sizeof(int));

    // Current offsets
    int full_offset = 0;
    int sign_mantissa_offset = 0;
    int tile_idx = 0;
    int median_offset_idx = 0;
    std::vector<int> global_high_freq_counts(num_global_tiles + 1, 0);
    std::vector<int> global_full_counts(num_global_tiles + 1, 0);

    max_high_freq_count = 0;
    max_full_count = 0;

    // Iterate over all global tiles
    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            int global_row_start = global_tile_m * tile_M_global;
            int global_col_start = global_tile_k * tile_K_global;
            int global_high_freq_count = 0;
            int global_full_count = 0;
            
            int median_high_freq_count = 0;
            int median_full_count = 0;
            
            // Store medium tile offsets
            (*TileOffsets_median)[median_offset_idx * 2] = 0;
            (*TileOffsets_median)[median_offset_idx * 2 + 1] = 0;
            median_offset_idx++;
            
            // Process medium tiles
            for (int median_tile_m = 0; median_tile_m < tile_M_global / tile_M_median; ++median_tile_m) {
                for (int median_tile_k = 0; median_tile_k < tile_K_global / tile_K_median; ++median_tile_k) {
                    int median_row_start = global_row_start + median_tile_m * tile_M_median;
                    int median_col_start = global_col_start + median_tile_k * tile_K_median;
                    
                    int local_median_high_freq = 0;
                    int local_median_full = 0;
                    
                    // Process 2x2 small tile groups
                    for (int local_tile_m_group = 0; local_tile_m_group < tile_M_median / tile_M; local_tile_m_group += 2) {
                        for (int local_tile_k_group = 0; local_tile_k_group < tile_K_median / tile_K; local_tile_k_group += 2) {
                            // Process 2x2 small tiles in column-major order
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

                                    // Process all elements in small tile
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
                                                
                                                // Find exponent position in high frequency list
                                                int exp_idx = -1;
                                                for (int e = 0; e < 7; e++) {
                                                    if (exponent == top_exponents[e]) {
                                                        exp_idx = e;
                                                        break;
                                                    }
                                                }
                                                
                                                bool is_high_freq = (exp_idx >= 0);
                                                
                                                if (is_high_freq) {
                                                    // High frequency exponent element
                                                    int bitmap_code = exp_idx + 1;  // 1-7
                                                    
                                                    // Set three bitmaps
                                                    tile_bitmap1 |= ((bitmap_code & 0x1) ? 1ULL << pos : 0);
                                                    tile_bitmap2 |= ((bitmap_code & 0x2) ? 1ULL << pos : 0);
                                                    tile_bitmap3 |= ((bitmap_code & 0x4) ? 1ULL << pos : 0);
                                                    
                                                    // Store sign+mantissa
                                                    uint8_t combined = ((sign & 0x1) << 7) | (mantissa & 0x7F);
                                                    (*sign_mantissa)[sign_mantissa_offset++] = combined;
                                                    
                                                    tile_high_freq_count++;
                                                    local_median_high_freq++;
                                                    global_high_freq_count++;
                                                } else {
                                                    // Non-high frequency exponent element
                                                    (*compressed_full)[full_offset++] = val;
                                                    
                                                    // Bitmap remains 000
                                                    tile_full_count++;
                                                    local_median_full++;
                                                    global_full_count++;
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
            
            // Add padding for high frequency elements (multiple of 16)
            int high_freq_padding = (16 - (global_high_freq_count % 16)) % 16;
            for (int p = 0; p < high_freq_padding; ++p) {
                (*sign_mantissa)[sign_mantissa_offset++] = 0;
            }
            global_high_freq_count += high_freq_padding;
            
            // Add padding for non-high frequency elements (multiple of 8)
            int full_padding = (8 - (global_full_count % 8)) % 8;
            for (int p = 0; p < full_padding; ++p) {
                (*compressed_full)[full_offset++] = __float2bfloat16(0.0f);
            }
            global_full_count += full_padding;
            
            // Record global tile counts
            global_high_freq_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_high_freq_count;
            global_full_counts[global_tile_m * num_global_tiles_K + global_tile_k + 1] = global_full_count;
            
            // Update max counts
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
    
    return num_global_tiles;
}

__host__ int InitBF16MatrixTripleBitmap_Reuse(
    __nv_bfloat16* A_bf16,
    int M,
    int K,
    int tile_M,          // 8
    int tile_M_median,   // 16
    int tile_M_global,   // 64
    int tile_K,          // 8
    int tile_K_median,   // 64
    int tile_K_global,   // 64
    const int* top_exponents,
    // Pre-allocated output buffers
    uint8_t* sign_mantissa,
    __nv_bfloat16* compressed_full,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    uint64_t* bitmap3,
    int* TileOffsets,
    int* TileOffsets_median,
    int* TileOffsets_global,
    // Pre-allocated temp workspace
    uint8_t* temp_sm,
    __nv_bfloat16* temp_full,
    int* gt_hf_count,
    int* gt_full_count,
    int* hf_offsets,
    int* full_offsets,
    // Outputs
    int& max_high_freq_count,
    int& max_full_count,
    int& out_total_hf,
    int& out_total_full)
{
    const int num_tiles_M = M / tile_M;
    const int num_tiles_K = K / tile_K;
    const int num_tiles = num_tiles_M * num_tiles_K;

    const int num_median_tiles_M = M / tile_M_median;
    const int num_median_tiles_K = K / tile_K_median;
    const int num_median_tiles = num_median_tiles_M * num_median_tiles_K;

    const int num_global_tiles_M = M / tile_M_global;
    const int num_global_tiles_K = K / tile_K_global;
    const int num_global_tiles = num_global_tiles_M * num_global_tiles_K;

    const int small_per_global = (tile_M_global / tile_M) * (tile_K_global / tile_K);
    const int median_per_global = (tile_M_global / tile_M_median) * (tile_K_global / tile_K_median);
    const int small_per_median = small_per_global / median_per_global;
    const int max_elem_per_gtile = tile_M_global * tile_K_global;

    const int max_sm_per_gtile = max_elem_per_gtile + 15;
    const int max_full_per_gtile = max_elem_per_gtile + 7;

    // Exponent lookup table
    uint8_t exp_to_idx[256];
    memset(exp_to_idx, 0xFF, 256);
    for (int e = 0; e < 7; e++)
        exp_to_idx[top_exponents[e]] = (uint8_t)e;

    // Precompute small tile layout within a global tile (flattened from 7-level nesting).
    const int num_median_m = tile_M_global / tile_M_median;
    const int num_median_k = tile_K_global / tile_K_median;
    const int num_small_m = tile_M_median / tile_M;
    const int num_small_k = tile_K_median / tile_K;
    struct TilePos { int row_off; int col_off; };
    TilePos tile_pos[256];
    {
        int tp_idx = 0;
        for (int mm = 0; mm < num_median_m; ++mm) {
            for (int mk = 0; mk < num_median_k; ++mk) {
                const int med_row = mm * tile_M_median;
                const int med_col = mk * tile_K_median;
                for (int mg = 0; mg < num_small_m; mg += 2) {
                    for (int kg = 0; kg < num_small_k; kg += 2) {
                        for (int j = 0; j < 2; ++j) {
                            for (int i = 0; i < 2; ++i) {
                                tile_pos[tp_idx].row_off = med_row + (mg + i) * tile_M;
                                tile_pos[tp_idx].col_off = med_col + (kg + j) * tile_K;
                                ++tp_idx;
                            }
                        }
                    }
                }
            }
        }
    }

    // ========== Pass 1: single-pass processing with flattened tile loop ==========
    #pragma omp parallel for schedule(dynamic) if(num_global_tiles >= 8)
    for (int gt = 0; gt < num_global_tiles; ++gt) {
        const int gm = gt / num_global_tiles_K;
        const int gk = gt % num_global_tiles_K;
        const int grow = gm * tile_M_global;
        const int gcol = gk * tile_K_global;

        uint8_t* my_sm = temp_sm + (size_t)gt * max_sm_per_gtile;
        __nv_bfloat16* my_full = temp_full + (size_t)gt * max_full_per_gtile;
        int sm_idx = 0;
        int full_idx = 0;

        const int base_tile_idx = gt * small_per_global;
        const int base_median_idx = gt * median_per_global;
        int cum_median_hf = 0;
        int cum_median_full = 0;
        int cur_median_hf = 0;
        int cur_median_full = 0;
        int local_median_idx = 1;

        TileOffsets_median[base_median_idx * 2] = 0;
        TileOffsets_median[base_median_idx * 2 + 1] = 0;

        for (int st = 0; st < small_per_global; ++st) {
            const int row_start = grow + tile_pos[st].row_off;
            const int col_start = gcol + tile_pos[st].col_off;

            uint64_t tb1 = 0, tb2 = 0, tb3 = 0;
            int t_hf = 0, t_full = 0;

            for (int ro = 0; ro < tile_M; ++ro) {
                const __nv_bfloat16* row_ptr = A_bf16 + (row_start + ro) * K + col_start;
                for (int co = 0; co < tile_K; ++co) {
                    const int pos = ro * tile_K + co;
                    const __nv_bfloat16 val = row_ptr[co];
                    const uint16_t bf16_bits = __bfloat16_as_ushort(val);
                    const uint8_t sign = (bf16_bits >> 15) & 0x1;
                    const uint8_t exponent = (bf16_bits >> 7) & 0xFF;
                    const uint8_t mantissa = bf16_bits & 0x7F;

                    const uint8_t eidx = exp_to_idx[exponent];
                    if (eidx != 0xFF) {
                        const int bitmap_code = eidx + 1;
                        tb1 |= ((bitmap_code & 0x1) ? 1ULL << pos : 0);
                        tb2 |= ((bitmap_code & 0x2) ? 1ULL << pos : 0);
                        tb3 |= ((bitmap_code & 0x4) ? 1ULL << pos : 0);
                        my_sm[sm_idx++] = ((sign & 0x1) << 7) | (mantissa & 0x7F);
                        t_hf++;
                    } else {
                        my_full[full_idx++] = val;
                        t_full++;
                    }
                }
            }

            const int tidx = base_tile_idx + st;
            bitmap1[tidx] = tb1;
            bitmap2[tidx] = tb2;
            bitmap3[tidx] = tb3;
            TileOffsets[tidx * 2] = t_hf;
            TileOffsets[tidx * 2 + 1] = t_full;

            cur_median_hf += t_hf;
            cur_median_full += t_full;

            if ((st % small_per_median) == (small_per_median - 1)) {
                const int mid = st / small_per_median;
                if (mid < median_per_global - 1) {
                    cum_median_hf += cur_median_hf;
                    cum_median_full += cur_median_full;
                    const int midx = base_median_idx + local_median_idx;
                    TileOffsets_median[midx * 2] = cum_median_hf;
                    TileOffsets_median[midx * 2 + 1] = cum_median_full;
                    local_median_idx++;
                }
                cur_median_hf = 0;
                cur_median_full = 0;
            }
        }

        const int hf_pad = (16 - (sm_idx % 16)) % 16;
        for (int p = 0; p < hf_pad; ++p)
            my_sm[sm_idx++] = 0;

        const int full_pad = (8 - (full_idx % 8)) % 8;
        for (int p = 0; p < full_pad; ++p)
            my_full[full_idx++] = __float2bfloat16(0.0f);

        gt_hf_count[gt] = sm_idx;
        gt_full_count[gt] = full_idx;
    }

    // ========== Pass 2: prefix sum ==========
    max_high_freq_count = 0;
    max_full_count = 0;
    hf_offsets[0] = 0;
    full_offsets[0] = 0;

    TileOffsets_global[0] = 0;
    TileOffsets_global[1] = 0;

    for (int i = 0; i < num_global_tiles; ++i) {
        hf_offsets[i + 1] = hf_offsets[i] + gt_hf_count[i];
        full_offsets[i + 1] = full_offsets[i] + gt_full_count[i];
        TileOffsets_global[(i + 1) * 2] = hf_offsets[i + 1];
        TileOffsets_global[(i + 1) * 2 + 1] = full_offsets[i + 1];
        if (gt_hf_count[i] > max_high_freq_count) max_high_freq_count = gt_hf_count[i];
        if (gt_full_count[i] > max_full_count) max_full_count = gt_full_count[i];
    }

    out_total_hf = hf_offsets[num_global_tiles];
    out_total_full = full_offsets[num_global_tiles];

    // ========== Pass 3: scatter from temp buffers to final arrays ==========
    #pragma omp parallel for schedule(static) if(num_global_tiles >= 8)
    for (int gt = 0; gt < num_global_tiles; ++gt) {
        if (gt_hf_count[gt] > 0)
            memcpy(sign_mantissa + hf_offsets[gt],
                   temp_sm + (size_t)gt * max_sm_per_gtile, gt_hf_count[gt]);
        if (gt_full_count[gt] > 0)
            memcpy(compressed_full + full_offsets[gt],
                   temp_full + (size_t)gt * max_full_per_gtile,
                   gt_full_count[gt] * sizeof(__nv_bfloat16));
    }

    return num_global_tiles;
}

// ---------------------------------------------------------------------------
// Precomputed shuffle masks for SIMD compress-store (indexed by 8-bit mask).
// Entry[mask] rearranges the bytes selected by `mask` to the front so that
// _mm_storel_epi64 writes them contiguously.
// ---------------------------------------------------------------------------
static __m128i s_compress_lut[256];
static bool    s_compress_lut_inited = false;

static void init_compress_lut() {
    if (s_compress_lut_inited) return;
    for (int mask = 0; mask < 256; ++mask) {
        alignas(16) uint8_t indices[16];
        int pos = 0;
        for (int b = 0; b < 8; ++b) {
            if (mask & (1 << b))
                indices[pos++] = (uint8_t)b;
        }
        for (int i = pos; i < 16; ++i)
            indices[i] = 0x80;
        s_compress_lut[mask] = _mm_load_si128((const __m128i*)indices);
    }
    s_compress_lut_inited = true;
}

// SIMD-accelerated version of InitBF16MatrixTripleBitmap_Reuse.
// The inner 8-element loop (one tile row) is replaced by SSE/SSSE3 vector ops:
//   - Range check replaces the 256-byte exp_to_idx LUT
//   - Bitmap bits extracted via packs + movemask
//   - sign_mantissa bytes compress-stored via pshufb + precomputed LUT
//   - Full (outlier) elements scattered with scalar code (typically 0-2/row)
// Pass 2 (prefix sum) and Pass 3 (memcpy scatter) are unchanged.
__host__ int InitBF16MatrixTripleBitmap_Reuse_SIMD(
    __nv_bfloat16* A_bf16,
    int M,
    int K,
    int tile_M,
    int tile_M_median,
    int tile_M_global,
    int tile_K,
    int tile_K_median,
    int tile_K_global,
    const int* top_exponents,
    // Pre-allocated output buffers
    uint8_t* sign_mantissa,
    __nv_bfloat16* compressed_full,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    uint64_t* bitmap3,
    int* TileOffsets,
    int* TileOffsets_median,
    int* TileOffsets_global,
    // Pre-allocated temp workspace
    uint8_t* temp_sm,
    __nv_bfloat16* temp_full,
    int* gt_hf_count,
    int* gt_full_count,
    int* hf_offsets,
    int* full_offsets,
    // Outputs
    int& max_high_freq_count,
    int& max_full_count,
    int& out_total_hf,
    int& out_total_full)
{
    init_compress_lut();

    const int num_global_tiles_M = M / tile_M_global;
    const int num_global_tiles_K = K / tile_K_global;
    const int num_global_tiles   = num_global_tiles_M * num_global_tiles_K;

    const int small_per_global  = (tile_M_global / tile_M) * (tile_K_global / tile_K);
    const int median_per_global = (tile_M_global / tile_M_median) * (tile_K_global / tile_K_median);
    const int small_per_median  = small_per_global / median_per_global;
    const int max_elem_per_gtile = tile_M_global * tile_K_global;

    const int max_sm_per_gtile   = max_elem_per_gtile + 15;
    const int max_full_per_gtile = max_elem_per_gtile + 7;

    // Top exponents are contiguous: [start_exp, start_exp + 6]
    const int start_exp = top_exponents[0];

    // Reinterpret BF16 input as uint16_t for SIMD loads
    const uint16_t* A_u16 = reinterpret_cast<const uint16_t*>(A_bf16);

    // Precompute small tile layout within a global tile (flattened from 7-level nesting)
    const int num_median_m = tile_M_global / tile_M_median;
    const int num_median_k = tile_K_global / tile_K_median;
    const int num_small_m  = tile_M_median / tile_M;
    const int num_small_k  = tile_K_median / tile_K;
    struct TilePos { int row_off; int col_off; };
    TilePos tile_pos[256];
    {
        int tp = 0;
        for (int mm = 0; mm < num_median_m; ++mm)
            for (int mk = 0; mk < num_median_k; ++mk) {
                const int mr = mm * tile_M_median;
                const int mc = mk * tile_K_median;
                for (int mg = 0; mg < num_small_m; mg += 2)
                    for (int kg = 0; kg < num_small_k; kg += 2)
                        for (int j = 0; j < 2; ++j)
                            for (int i = 0; i < 2; ++i) {
                                tile_pos[tp].row_off = mr + (mg + i) * tile_M;
                                tile_pos[tp].col_off = mc + (kg + j) * tile_K;
                                ++tp;
                            }
            }
    }

    // SSE constant vectors
    const __m128i v_start  = _mm_set1_epi16((short)start_exp);
    const __m128i v_seven  = _mm_set1_epi16(7);
    const __m128i v_neg1   = _mm_set1_epi16(-1);
    const __m128i v_one    = _mm_set1_epi16(1);
    const __m128i v_0xFF   = _mm_set1_epi16(0x00FF);
    const __m128i v_0x80   = _mm_set1_epi16(0x0080);
    const __m128i v_0x7F   = _mm_set1_epi16(0x007F);
    const __m128i v_zero   = _mm_setzero_si128();

    // ========== Pass 1: SIMD-accelerated processing ==========
    #pragma omp parallel for schedule(dynamic) if(num_global_tiles >= 8)
    for (int gt = 0; gt < num_global_tiles; ++gt) {
        const int gm   = gt / num_global_tiles_K;
        const int gk   = gt % num_global_tiles_K;
        const int grow  = gm * tile_M_global;
        const int gcol  = gk * tile_K_global;

        uint8_t*  my_sm   = temp_sm + (size_t)gt * max_sm_per_gtile;
        uint16_t* my_full = reinterpret_cast<uint16_t*>(temp_full + (size_t)gt * max_full_per_gtile);
        int sm_idx   = 0;
        int full_idx = 0;

        const int base_tile_idx   = gt * small_per_global;
        const int base_median_idx = gt * median_per_global;
        int cum_median_hf   = 0;
        int cum_median_full = 0;
        int cur_median_hf   = 0;
        int cur_median_full = 0;
        int local_median_idx = 1;

        TileOffsets_median[base_median_idx * 2]     = 0;
        TileOffsets_median[base_median_idx * 2 + 1] = 0;

        for (int st = 0; st < small_per_global; ++st) {
            const int row_start = grow + tile_pos[st].row_off;
            const int col_start = gcol + tile_pos[st].col_off;

            uint64_t tb1 = 0, tb2 = 0, tb3 = 0;
            int t_hf = 0, t_full = 0;

            for (int ro = 0; ro < tile_M; ++ro) {
                const uint16_t* row_ptr = A_u16 + (row_start + ro) * K + col_start;

                // Load 8 × BF16 (uint16_t)
                __m128i bits = _mm_loadu_si128((const __m128i*)row_ptr);

                // Exponent: (bits >> 7) & 0xFF
                __m128i exp_vec = _mm_and_si128(_mm_srli_epi16(bits, 7), v_0xFF);

                // Range check: sub = exp - start; is_hf = (sub >= 0) && (sub < 7)
                __m128i sub_vec  = _mm_sub_epi16(exp_vec, v_start);
                __m128i ge_zero  = _mm_cmpgt_epi16(sub_vec, v_neg1);
                __m128i lt_seven = _mm_cmpgt_epi16(v_seven, sub_vec);
                __m128i is_hf    = _mm_and_si128(ge_zero, lt_seven);

                // 8-bit high-freq mask
                int hf_mask = _mm_movemask_epi8(_mm_packs_epi16(is_hf, v_zero)) & 0xFF;
                int n_hf    = __builtin_popcount(hf_mask);

                // bitmap_code = (sub + 1) masked to hf-only lanes
                __m128i bc = _mm_and_si128(_mm_add_epi16(sub_vec, v_one), is_hf);

                // Extract bitmap bits and merge into tb1/tb2/tb3
                __m128i b0 = _mm_cmpeq_epi16(_mm_and_si128(bc, v_one), v_one);
                int tb1_bits = _mm_movemask_epi8(_mm_packs_epi16(b0, v_zero)) & 0xFF;

                __m128i b1 = _mm_cmpeq_epi16(_mm_and_si128(_mm_srli_epi16(bc, 1), v_one), v_one);
                int tb2_bits = _mm_movemask_epi8(_mm_packs_epi16(b1, v_zero)) & 0xFF;

                __m128i b2 = _mm_cmpeq_epi16(_mm_and_si128(_mm_srli_epi16(bc, 2), v_one), v_one);
                int tb3_bits = _mm_movemask_epi8(_mm_packs_epi16(b2, v_zero)) & 0xFF;

                tb1 |= (uint64_t)tb1_bits << (ro * 8);
                tb2 |= (uint64_t)tb2_bits << (ro * 8);
                tb3 |= (uint64_t)tb3_bits << (ro * 8);

                // sign_mantissa: sm = ((bits >> 8) & 0x80) | (bits & 0x7F)
                __m128i sm16 = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16(bits, 8), v_0x80),
                    _mm_and_si128(bits, v_0x7F));
                __m128i sm8 = _mm_packus_epi16(sm16, v_zero);

                // Compress-store: shuffle hf bytes to front, then store
                __m128i compressed = _mm_shuffle_epi8(sm8, s_compress_lut[hf_mask]);
                _mm_storel_epi64((__m128i*)(my_sm + sm_idx), compressed);
                sm_idx += n_hf;
                t_hf   += n_hf;

                // Full elements (scalar, typically 0-2 per row)
                int fm = (~hf_mask) & 0xFF;
                if (fm) {
                    uint16_t row_vals[8];
                    _mm_storeu_si128((__m128i*)row_vals, bits);
                    while (fm) {
                        int b = __builtin_ctz(fm);
                        my_full[full_idx++] = row_vals[b];
                        t_full++;
                        fm &= fm - 1;
                    }
                }
            }

            const int tidx = base_tile_idx + st;
            bitmap1[tidx] = tb1;
            bitmap2[tidx] = tb2;
            bitmap3[tidx] = tb3;
            TileOffsets[tidx * 2]     = t_hf;
            TileOffsets[tidx * 2 + 1] = t_full;

            cur_median_hf   += t_hf;
            cur_median_full += t_full;

            if ((st % small_per_median) == (small_per_median - 1)) {
                const int mid = st / small_per_median;
                if (mid < median_per_global - 1) {
                    cum_median_hf   += cur_median_hf;
                    cum_median_full += cur_median_full;
                    const int midx = base_median_idx + local_median_idx;
                    TileOffsets_median[midx * 2]     = cum_median_hf;
                    TileOffsets_median[midx * 2 + 1] = cum_median_full;
                    local_median_idx++;
                }
                cur_median_hf   = 0;
                cur_median_full = 0;
            }
        }

        const int hf_pad = (16 - (sm_idx % 16)) % 16;
        for (int p = 0; p < hf_pad; ++p)
            my_sm[sm_idx++] = 0;

        const int full_pad = (8 - (full_idx % 8)) % 8;
        for (int p = 0; p < full_pad; ++p)
            my_full[full_idx++] = 0;

        gt_hf_count[gt]  = sm_idx;
        gt_full_count[gt] = full_idx;
    }

    // ========== Pass 2: prefix sum ==========
    max_high_freq_count = 0;
    max_full_count      = 0;
    hf_offsets[0]   = 0;
    full_offsets[0] = 0;

    TileOffsets_global[0] = 0;
    TileOffsets_global[1] = 0;

    for (int i = 0; i < num_global_tiles; ++i) {
        hf_offsets[i + 1]   = hf_offsets[i]   + gt_hf_count[i];
        full_offsets[i + 1] = full_offsets[i]  + gt_full_count[i];
        TileOffsets_global[(i + 1) * 2]     = hf_offsets[i + 1];
        TileOffsets_global[(i + 1) * 2 + 1] = full_offsets[i + 1];
        if (gt_hf_count[i] > max_high_freq_count) max_high_freq_count = gt_hf_count[i];
        if (gt_full_count[i] > max_full_count) max_full_count = gt_full_count[i];
    }

    out_total_hf   = hf_offsets[num_global_tiles];
    out_total_full = full_offsets[num_global_tiles];

    // ========== Pass 3: scatter from temp buffers to final arrays ==========
    #pragma omp parallel for schedule(static) if(num_global_tiles >= 8)
    for (int gt = 0; gt < num_global_tiles; ++gt) {
        if (gt_hf_count[gt] > 0)
            memcpy(sign_mantissa + hf_offsets[gt],
                   temp_sm + (size_t)gt * max_sm_per_gtile, gt_hf_count[gt]);
        if (gt_full_count[gt] > 0)
            memcpy(compressed_full + full_offsets[gt],
                   temp_full + (size_t)gt * max_full_per_gtile,
                   gt_full_count[gt] * sizeof(__nv_bfloat16));
    }

    return num_global_tiles;
}

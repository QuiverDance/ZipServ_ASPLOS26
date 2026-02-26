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
    int tile_M,
    int tile_M_median,
    int tile_M_global,
    int tile_K,
    int tile_K_median,
    int tile_K_global,
    const int* top_exponents,
    uint8_t* sign_mantissa,
    __nv_bfloat16* compressed_full,
    uint64_t* bitmap1,
    uint64_t* bitmap2,
    uint64_t* bitmap3,
    int* TileOffsets,
    int* TileOffsets_median,
    int* TileOffsets_global,
    uint8_t* temp_sm,
    __nv_bfloat16* temp_full,
    int* gt_hf_count,
    int* gt_full_count,
    int* hf_offsets,
    int* full_offsets,
    int& max_high_freq_count,
    int& max_full_count,
    int& out_total_hf,
    int& out_total_full)
{
    if (A_bf16 == nullptr || top_exponents == nullptr ||
        sign_mantissa == nullptr || compressed_full == nullptr ||
        bitmap1 == nullptr || bitmap2 == nullptr || bitmap3 == nullptr ||
        TileOffsets == nullptr || TileOffsets_median == nullptr || TileOffsets_global == nullptr ||
        temp_sm == nullptr || temp_full == nullptr ||
        gt_hf_count == nullptr || gt_full_count == nullptr ||
        hf_offsets == nullptr || full_offsets == nullptr) {
        return -1;
    }
    if (M <= 0 || K <= 0 || tile_M <= 0 || tile_M_median <= 0 || tile_M_global <= 0 ||
        tile_K <= 0 || tile_K_median <= 0 || tile_K_global <= 0) {
        return -1;
    }

    const int num_tiles_M = M / tile_M;
    const int num_tiles_K = K / tile_K;
    const int num_tiles = num_tiles_M * num_tiles_K;

    const int num_median_tiles_M = M / tile_M_median;
    const int num_median_tiles_K = K / tile_K_median;
    const int num_median_tiles = num_median_tiles_M * num_median_tiles_K;

    const int num_global_tiles_M = M / tile_M_global;
    const int num_global_tiles_K = K / tile_K_global;
    const int num_global_tiles = num_global_tiles_M * num_global_tiles_K;
    if (num_global_tiles <= 0) {
        return -1;
    }

    (void)temp_sm;
    (void)temp_full;

    memset(compressed_full, 0, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16));
    memset(sign_mantissa, 0, static_cast<size_t>(M) * static_cast<size_t>(K));
    memset(bitmap1, 0, static_cast<size_t>(num_tiles) * sizeof(uint64_t));
    memset(bitmap2, 0, static_cast<size_t>(num_tiles) * sizeof(uint64_t));
    memset(bitmap3, 0, static_cast<size_t>(num_tiles) * sizeof(uint64_t));
    memset(TileOffsets, 0, static_cast<size_t>(num_tiles) * 2 * sizeof(int));
    memset(TileOffsets_median, 0, static_cast<size_t>(num_median_tiles) * 2 * sizeof(int));
    memset(TileOffsets_global, 0, static_cast<size_t>(num_global_tiles + 1) * 2 * sizeof(int));
    memset(gt_hf_count, 0, static_cast<size_t>(num_global_tiles) * sizeof(int));
    memset(gt_full_count, 0, static_cast<size_t>(num_global_tiles) * sizeof(int));
    memset(hf_offsets, 0, static_cast<size_t>(num_global_tiles + 1) * sizeof(int));
    memset(full_offsets, 0, static_cast<size_t>(num_global_tiles + 1) * sizeof(int));

    int full_offset = 0;
    int sign_mantissa_offset = 0;
    int tile_idx = 0;
    int median_offset_idx = 0;

    max_high_freq_count = 0;
    max_full_count = 0;
    out_total_hf = 0;
    out_total_full = 0;

    for (int global_tile_m = 0; global_tile_m < num_global_tiles_M; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_global_tiles_K; ++global_tile_k) {
            const int global_row_start = global_tile_m * tile_M_global;
            const int global_col_start = global_tile_k * tile_K_global;
            int global_high_freq_count = 0;
            int global_full_count = 0;

            int median_high_freq_count = 0;
            int median_full_count = 0;

            TileOffsets_median[median_offset_idx * 2] = 0;
            TileOffsets_median[median_offset_idx * 2 + 1] = 0;
            median_offset_idx++;

            for (int median_tile_m = 0; median_tile_m < tile_M_global / tile_M_median; ++median_tile_m) {
                for (int median_tile_k = 0; median_tile_k < tile_K_global / tile_K_median; ++median_tile_k) {
                    const int median_row_start = global_row_start + median_tile_m * tile_M_median;
                    const int median_col_start = global_col_start + median_tile_k * tile_K_median;

                    int local_median_high_freq = 0;
                    int local_median_full = 0;

                    for (int local_tile_m_group = 0; local_tile_m_group < tile_M_median / tile_M; local_tile_m_group += 2) {
                        for (int local_tile_k_group = 0; local_tile_k_group < tile_K_median / tile_K; local_tile_k_group += 2) {
                            for (int j = 0; j < 2; ++j) {
                                for (int i = 0; i < 2; ++i) {
                                    const int local_tile_k = local_tile_k_group + j;
                                    const int local_tile_m = local_tile_m_group + i;

                                    const int col_start = median_col_start + local_tile_k * tile_K;
                                    const int row_start = median_row_start + local_tile_m * tile_M;

                                    uint64_t tile_bitmap1 = 0;
                                    uint64_t tile_bitmap2 = 0;
                                    uint64_t tile_bitmap3 = 0;
                                    int tile_high_freq_count = 0;
                                    int tile_full_count = 0;

                                    for (int row_offset = 0; row_offset < tile_M; ++row_offset) {
                                        for (int col_offset = 0; col_offset < tile_K; ++col_offset) {
                                            const int row = row_start + row_offset;
                                            const int col = col_start + col_offset;
                                            const int pos = row_offset * tile_K + col_offset;

                                            if (row < M && col < K) {
                                                const __nv_bfloat16 val = A_bf16[row * K + col];

                                                const uint16_t bf16_bits = __bfloat16_as_ushort(val);
                                                const uint8_t sign = (bf16_bits >> 15) & 0x1;
                                                const uint8_t exponent = (bf16_bits >> 7) & 0xFF;
                                                const uint8_t mantissa = bf16_bits & 0x7F;

                                                int exp_idx = -1;
                                                for (int e = 0; e < 7; ++e) {
                                                    if (exponent == top_exponents[e]) {
                                                        exp_idx = e;
                                                        break;
                                                    }
                                                }

                                                if (exp_idx >= 0) {
                                                    const int bitmap_code = exp_idx + 1;
                                                    tile_bitmap1 |= ((bitmap_code & 0x1) ? (1ULL << pos) : 0);
                                                    tile_bitmap2 |= ((bitmap_code & 0x2) ? (1ULL << pos) : 0);
                                                    tile_bitmap3 |= ((bitmap_code & 0x4) ? (1ULL << pos) : 0);

                                                    const uint8_t combined = ((sign & 0x1) << 7) | (mantissa & 0x7F);
                                                    sign_mantissa[sign_mantissa_offset++] = combined;

                                                    tile_high_freq_count++;
                                                    local_median_high_freq++;
                                                    global_high_freq_count++;
                                                } else {
                                                    compressed_full[full_offset++] = val;
                                                    tile_full_count++;
                                                    local_median_full++;
                                                    global_full_count++;
                                                }
                                            }
                                        }
                                    }

                                    bitmap1[tile_idx] = tile_bitmap1;
                                    bitmap2[tile_idx] = tile_bitmap2;
                                    bitmap3[tile_idx] = tile_bitmap3;
                                    TileOffsets[tile_idx * 2] = tile_high_freq_count;
                                    TileOffsets[tile_idx * 2 + 1] = tile_full_count;
                                    ++tile_idx;
                                }
                            }
                        }
                    }

                    if (median_tile_m < (tile_M_global / tile_M_median - 1) ||
                        median_tile_k < (tile_K_global / tile_K_median - 1)) {
                        median_high_freq_count += local_median_high_freq;
                        median_full_count += local_median_full;

                        TileOffsets_median[median_offset_idx * 2] = median_high_freq_count;
                        TileOffsets_median[median_offset_idx * 2 + 1] = median_full_count;
                        median_offset_idx++;
                    }
                }
            }

            const int high_freq_padding = (16 - (global_high_freq_count % 16)) % 16;
            for (int p = 0; p < high_freq_padding; ++p) {
                sign_mantissa[sign_mantissa_offset++] = 0;
            }
            global_high_freq_count += high_freq_padding;

            const int full_padding = (8 - (global_full_count % 8)) % 8;
            for (int p = 0; p < full_padding; ++p) {
                compressed_full[full_offset++] = __float2bfloat16(0.0f);
            }
            global_full_count += full_padding;

            const int global_idx = global_tile_m * num_global_tiles_K + global_tile_k;
            gt_hf_count[global_idx] = global_high_freq_count;
            gt_full_count[global_idx] = global_full_count;
            hf_offsets[global_idx + 1] = global_high_freq_count;
            full_offsets[global_idx + 1] = global_full_count;

            if (global_high_freq_count > max_high_freq_count) {
                max_high_freq_count = global_high_freq_count;
            }
            if (global_full_count > max_full_count) {
                max_full_count = global_full_count;
            }
        }
    }

    TileOffsets_global[0] = 0;
    TileOffsets_global[1] = 0;
    for (int i = 1; i <= num_global_tiles; ++i) {
        hf_offsets[i] += hf_offsets[i - 1];
        full_offsets[i] += full_offsets[i - 1];
        TileOffsets_global[i * 2] = hf_offsets[i];
        TileOffsets_global[i * 2 + 1] = full_offsets[i];
    }

    out_total_hf = hf_offsets[num_global_tiles];
    out_total_full = full_offsets[num_global_tiles];
    return num_global_tiles;
}

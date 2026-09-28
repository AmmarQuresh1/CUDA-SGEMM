#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
__launch_bounds__(max threads per block, min blocks per multiprocessor)
BM * BN is the block size in C and we divide by TM * TN to give each thread a tile in the block.
This will reduce register pressure as we increase arithmetic intensity (each thread calculates more results).
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN> 
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) 
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A, 
    const float *B, float beta, float *C) {
        const uint cRow = blockIdx.y;
        const uint cCol = blockIdx.x;

        const uint totalResultsBlocktile = BM * BN;
        const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

        // Results per block / results per thread == threads per block
        assert(numThreadsBlocktile == blockDim.x);
        
        const int threadCol = threadIdx.x % (BN / TN);
        const int threadRow = threadIdx.x / (BN / TN);

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        A += cRow * BM * K;
        B += cCol * BN;
        C += cRow * BM * N + cCol * BN;

        const uint innerRowA = threadIdx.x / BK;
        const uint innerColA = threadIdx.x % BK;

        const uint strideA = numThreadsBlocktile / BK;
        const uint innerRowB = threadIdx.x / BN;
        const uint innerColB = threadIdx.x % BN;
        const uint strideB = numThreadsBlocktile / BN;

        float threadResults[TM * TN] = {0.0};
        // As and Bs register caches
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                As[(innerRowA + loadOffset) * BK + innerColA] = 
                    A[(innerRowA + loadOffset) * K + innerColA];
            }
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                Bs[(innerRowB + loadOffset) * BN + innerColB] = 
                    B[(innerRowB + loadOffset) * N + innerColA];
            }
            __syncthreads();

            A += BK;
            B += BK * N;

            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                for (uint i = 0; i < TM; ++i) {
                    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
                }
                for (uint i = 0; i < TN; ++i) {
                    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
                }
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[resIdxM * TN + resIdxN] += 
                            regM[resIdxM] * regN[resIdxN];
                    }
                }
            }
            __syncthreads();
        }
        

        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = 
                    alpha * threadResults[resIdxM * TN + resIdxN] + 
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
            }
        }

    }

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
Default values:
const uint BM = 64; 
const uint BN = 64; 
const uint BK = 8;
const uint TM = 8;
*/

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // output block we want to compute in this threadblock
    // Swapping indices keeps blocks sequential
    // accessing columns of B sequentially while sharing the same columns of A.
    // Better spacial locality, and greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, 32 is the column dim
    // threadCol -> matrix B, each thread calculates a col of results
    // threadRow -> matrix A, each thread calculates a row of results
    // both used to find result in C
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for current blocktile in SMEM
    __shared__ float As[BM * BK]; 
    __shared__ float Bs[BK * BN];

    // advance pointers to the starting positions
    A += cRow * BM * K;
    B += cCol * BN; 
    C += cRow * BM * N + cCol * BN; 

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    // warp coalescing  
    const uint innerColA = threadIdx.x % BK; 
    const uint innerRowA = threadIdx.x / BK; 
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN; 

    // thread-local cache in registerfile for the row/col it computes
    float threadResults[TM] = {0.0};

    // the outer loop advances A along the columns and B along rows 
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Populate SMEM caches, each thread populates 1 elem
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BM + innerColB] = B[innerRowB * N + innerColB];
         __syncthreads();

        // advance blockfile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // loop over the outer rows of A so that B's element can be reused and cached
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB; // select how far down the row should go, * by BK to get the actual flattened loc + column offset 
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

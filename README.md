# CUDA SGEMM 

Implementing Simon Boehm's CUDA Matmul Kernel Worklog, starting with naive kernel and applying optimisations to get within 95% of cuBLAS.

## Build instructions
```bash
mkdir build && cd build
cmake ..
cmake --build . 
cd ..
ln -s build/compile_commands.json
./sgemm {kernel_number - 0 for cuBLAS, 1 for naive, etc.}
```

## Profiling
```bash
// profiles once for 4096 size kernel
ncu --set basic --section WarpStateStats --section SchedulerStats --section InstructionStats --launch-count 1 --launch-skip 300 -o profiles/{name} -f ./sgemm {kernel_num}

// read profiler report
ncu --import profiles/{report} > profiles/{name}.txt
```

## Results
**Benchmarked using a 3060Ti:**
| **Property** | **RTX 3060 Ti (Ampere, compute 8.6)** |
| --- | --- |
| Number of SMs | 38 |
| Max threads per SM | 1536 |
| Max registers per SM | 65,536 |
| Hardware max L1/Shared Memory per SM | 128 KB |
| Max configurable shared memory per SM | 0, 8, 16, 32, 64 or 100 KB per SM |
| Max warps per SM | 48 |


The general matrix multiplication formula is defined as: $C = \alpha \cdot (A \cdot B) + \beta \cdot C$:
- Matrices: $A$, $B$ and $C$, randomised and increasing in size (128, 256, 512, 1024, 2048, 4096)
- Scalar constants: $\alpha$ = 0.5 and $\beta$ = 3.0

Improved performance for kernels 1-3  (~1.95x speedup kernel 1) by changing threads per block from 1024 to 256. 
This better utilises the 3060Ti's capability of 1536 threads per SM.

### cuBLAS (Reference) 
**10691.2 GFLOPs/s**

L1/TEX Cache Throughput (57.95%) & Memory Throughput (57.84%). Achieved Occupancy (33.14%).
Compute-bound reference.

```bash
Running kernel 0 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000006) s, performance: (  699.0) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000009) s, performance: ( 3668.1) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000041) s, performance: ( 6508.0) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.000226) s, performance: ( 9487.7) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.001655) s, performance: (10380.3) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.012855) s, performance: (10691.2) GFLOPS. size: (4096).
```

### Naive - Kernel 1 
**282.0 GFLOPs/s**

**Performance relative to cuBLAS - 2.63%**

L1/TEX Cache Throughput (99.93%) and Memory Throughput (99.81%) maxed, memory-bound instead of compute-bound.
Achieved Occupancy (88.13%).

Uncoalesced access wastes loaded data from GMEM as the warp does not use the entire fetched contiguous sector.

```bash
Running kernel 1 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000023) s, performance: (  186.2) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000146) s, performance: (  230.1) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.001090) s, performance: (  246.2) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.007944) s, performance: (  270.3) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.061056) s, performance: (  281.4) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.487344) s, performance: (  282.0) GFLOPS. size: (4096).
```

### Global Memory Coalescing - Kernel 2 
**931.1 GFLOPs/s**

**Performance relative to cuBLAS - 8.71%**

L1/TEX Cache Throughput (85.13%) & Memory Throughput (85%) still high, memory-bound.
Achieved Occupancy (99.86%).

Kernel 2's warps fully utilise the fetched sector from GMEM but each thread still reads from global memory which can be improved by caching into SMEM. 

```bash
Running kernel 2 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000009) s, performance: (  477.3) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000043) s, performance: (  772.2) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000284) s, performance: (  946.6) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.002140) s, performance: ( 1003.5) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.017167) s, performance: ( 1000.8) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.147613) s, performance: (  931.1) GFLOPS. size: (4096).
```

### Shared Memory Cache-Blocking - Kernel 3 
**1552.6 GFLOPs/s**

**Performance relative to cuBLAS - 14.52%**

L1/TEX Cache Throughput (97.52%) & Memory Throughput (97.50%), still memory-bound.
Achieved Occupancy (99.82%).

Shared memory: 2048B/Block + 1024B/Block for CUDA runtime usage = 3072B/Block
Theoretical maximum blocks per SM: 101376B / 3072B = 33 blocks
Configured SMEM maximum blocks per SM: 32,768B / 3072B = 10 blocks 

Threads: 256 threads per block, max 1536 threads per SM -> up to 6 blocks

Registers: 40 regs per thread * 32 threads per warp = 1280 regs per warp. Register allocation granularity 256 regs per warp, 256*5 = 1280 so no need to round up. 256 threads per block, 256/32 = 8 warps per block, so 8 * 1280 = 10,240 regs per block. Utilised 61,440 regs per SM as up to 6 blocks per SM. 

The kernel is limited by the number of threads per block and registers per thread as we cannot load more than 6 blocks per SM. The number of active warps is 48 (warps per block * max blocks per SM), this is the same as the 48 max active warps giving an occupancy of ~100%.

Each warp spends 24.1 cycles (out of the 53.10 cycles per issued instruction) being stalled waiting for the MIO instruction queue to not be full. The scheduler allocated 11.98 warps out of the maximum 12 but only 1.06 warps were eligible per cycle with No Eligible at 77.45%. Warps spend almost half their cycles waiting to issue, with 11.98 active warps the scheduler will wait 4.4 cycles before it can issue the next warp's instructions. The bottleneck for this kernel then comes from the high usage of SMEM instructions. 

```bash
Running kernel 3 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000009) s, performance: (  493.4) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000029) s, performance: ( 1151.5) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000201) s, performance: ( 1337.8) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.001552) s, performance: ( 1383.6) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.011109) s, performance: ( 1546.5) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.088520) s, performance: ( 1552.6) GFLOPS. size: (4096).
```

### 1D Blocktiling for Calculating Multiple Results per Thread - Kernel 4 
**4372.3 GFLOPs/s**

**Performance relative to cuBLAS - 40.90%**

L1/TEX Cache Throughput (81.71%) & Memory Throughput (81.50%), still memory-bound.

Achieved Occupancy (66.36%), limited by number of registers. On threads alone, 512 threads per block would mean three blocks fit into the 3060Ti's SM. However registers per thread is at 44, so 1,408 registers per warp (rounds up to 1536 regs/warp due to allocation granularity) which gives 24,576 registers per block. Dividing the max registers per SM (65,536) by regs/block gives ~2.67 blocks, only 2 whole blocks fit which is why the theoretical occupancy is 66.67%. Performance still improves as each thread now computes 8 results instead of 1. 

Stall MIO throttle is improved but still not good, the scheduler can only issue an instruction every 2.6 cycles instead of the 4.4 cycles in kernel 3. To lower this further each thread must compute even more results per thread, which increases arithmetic intensity (FLOPs executed per byte transferred (load + store)). 

```bash
Running kernel 4 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000014) s, performance: (  289.8) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000026) s, performance: ( 1278.8) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000087) s, performance: ( 3088.5) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.000580) s, performance: ( 3703.6) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.004221) s, performance: ( 4069.7) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.031434) s, performance: ( 4372.3) GFLOPS. size: (4096).
```

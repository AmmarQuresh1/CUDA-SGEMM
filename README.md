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
ncu --set basic --launch-count 1 --launch-skip 300 -o profiles/{name} -f ./sgemm {kernel_num}

// read profiler report
ncu --import profiles/{report} > profiles/{name}.txt
```

## Results
- Benchmarked using a 3060Ti
- The general matrix multiplication formula is defined as: $C = \alpha \cdot (A \cdot B) + \beta \cdot C$
    - Matrices: $A$, $B$ and $C$, randomised and increasing in size (128, 256, 512, 1024, 2048, 4096)
    - Scalar constants: $\alpha$ = 0.5 and $\beta$ = 3.0
- Improved performance up to ~1.95x (kernel 1) by changing threads per block from 1024 to 256. This better utilises the 3060Ti's capability of 1536 threads per SM.

### cuBLAS (Reference)

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
**Performance relative to cuBLAS - 2.63%**

L1/TEX Cache Throughput (99.93%) and Memory Throughput (99.81%) maxed, memory-bound instead of compute-bound.
Achieved Occupancy (88.13%).

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
**Performance relative to cuBLAS - 8.71%**

L1/TEX Cache Throughput (85.13%) & Memory Throughput (85%) still high, memory-bound.
Achieved Occupancy (99.86%).

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
**Performance relative to cuBLAS - 14.52%**

L1/TEX Cache Throughput (97.52%) & Memory Throughput (97.50%), still memory-bound.
Achieved Occupancy (99.82%).


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

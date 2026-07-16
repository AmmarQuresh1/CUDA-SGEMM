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

## Results
- Benchmarked using a 3060Ti
- The general matrix multiplication formula is defined as: $C = \alpha \cdot (A \cdot B) + \beta \cdot C$
    - Matrices: $A$, $B$ and $C$, randomised and increasing in size (128, 256, 512, 1024, 2048, 4096)
    - Scalar constants: $\alpha$ = 0.5 and $\beta$ = 3.0

### cuBLASS (Reference)
```bash
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000707) s, performance: (    5.9) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000010) s, performance: ( 3295.1) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.000043) s, performance: ( 6210.3) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.000227) s, performance: ( 9468.8) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.001639) s, performance: (10482.1) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.012720) s, performance: (10805.3) GFLOPS. size: (4096).
```

### Naive - Kernel 1 
**Performance relative to cuBLAS - 1.34%**
```bash
Running kernel 1 on device 0.
Max size: 4096
dimensions(m=n=k) 128, alpha: 0.5, beta: 3
Average elapsed time: (0.000081) s, performance: (   52.1) GFLOPS. size: (128).
dimensions(m=n=k) 256, alpha: 0.5, beta: 3
Average elapsed time: (0.000314) s, performance: (  106.8) GFLOPS. size: (256).
dimensions(m=n=k) 512, alpha: 0.5, beta: 3
Average elapsed time: (0.002173) s, performance: (  123.5) GFLOPS. size: (512).
dimensions(m=n=k) 1024, alpha: 0.5, beta: 3
Average elapsed time: (0.015147) s, performance: (  141.8) GFLOPS. size: (1024).
dimensions(m=n=k) 2048, alpha: 0.5, beta: 3
Average elapsed time: (0.118571) s, performance: (  144.9) GFLOPS. size: (2048).
dimensions(m=n=k) 4096, alpha: 0.5, beta: 3
Average elapsed time: (0.947552) s, performance: (  145.0) GFLOPS. size: (4096).
```

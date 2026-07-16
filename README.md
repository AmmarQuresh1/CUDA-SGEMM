# CUDA SGEMM 

- Following Simon Boehm's CUDA Matmul Kernel Worklog, starting with naive kernel and applying optimisations to get within 95% of cuBLAS.
- Implemented on 3060Ti.

## Build instructions
```bash
mkdir build && cd build
cmake ..
cmake --build . 
./sgemm {kernel_number - 0 for cuBLAS, 1 for naive, etc.}
```

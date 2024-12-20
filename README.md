
# Parallel 2D Laplace Equation Solver

## Overview
This project implements a parallel solver for the 2D Laplace equation using MPI and CUDA. Two numerical methods, Jacobi iteration and Red-Black Gauss-Seidel iteration, are implemented to demonstrate parallel computing on both CPU and GPU.

The program implements:  
- **CUDA** for accelerating computational kernels.  
- **OpenMPI** for parallel communication across multiple processes.  
- A **Makefile** to streamline the build process.  

The program supports two iterative methods for solving the 2D Laplace equation:  
- **Jacobi Iteration**  
- **Red-Black Gauss-Seidel Iteration**  

---

## Test Environment  

### Hardware  
- **CPU**: Intel Core i9-9980HK (8 cores, 16 threads)  
- **GPU**: NVIDIA Tesla P4 (8 GB DRAM)  

### Software  
- **Operating System**: WSL Ubuntu 24.04.1 LTS  
- **Compiler**: GCC/G++ 11  
- **MPI Implementation**: OpenMPI 4.1.6  
- **CUDA Toolkit**: Version 11.8  

---

## File Descriptions  

### Source Files  
- `main.c`: Implements the main program logic for solving the 2D Laplace equation using MPI.  
- `cuda_jacobi.cu`: CUDA kernel for Jacobi iteration.  
- `cuda_gauss.cu`: CUDA kernel for Red-Black Gauss-Seidel iteration.  

### Build System  
- `Makefile`: Automates the compilation of the source files into an executable.  

### Intermediate Files  
- `main.o`: Object file for `main.c`.  
- `cuda_jacobi.o`: Object file for `cuda_jacobi.cu`.  
- `cuda_gauss.o`: Object file for `cuda_gauss.cu`.  

### Executable  
- `laplace_solver`: The final executable file.  

---

## Compilation
To compile the program, simply run:
```bash
make
```

## Execution
The executable `laplace_solver` accepts two command-line arguments:
1. **Matrix Dimension**: Size of the 2D grid (e.g., 1024).
2. **Iteration Method**: Numerical method to use for solving:
   - `0`: Jacobi iteration
   - `1`: Red-Black Gauss-Seidel iteration
# Example
```bash
mpirun -np 4 ./laplace_solver 1024 0
```
This command runs the solver with 4 MPI processes on a 1024×1024 problem using the Jacobi iteration method.

---


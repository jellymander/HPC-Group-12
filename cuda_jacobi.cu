#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 32

// CUDA 内核：执行 Jacobi 迭代的更新
__global__ void jacobi_update(float *u, float *new_u, int Nx, int Ny,int gridx, int gridy, int restx, int resty, float *blk_diff) { 
    // 数组 u 和 new_u 是 MPI 划分到单个进程上的数组，维度为 Nx * Ny
    // 数组 blk_diff ( cuda 主程序中的 d_local_diff) 存储 Grid 内 所有 Block 的 最大 diff

    __shared__ float thread_diff [BLOCK_SIZE * BLOCK_SIZE]; // 共享内存，存储一个 Block 内 所有 Threads 的 diff
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;  // 数组 u 访问的分量 1 （从 i = 1 开始）
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;  // 数组 u 访问的分量 2 （从 j = 1 开始）
	if(blockIdx.x==gridx-1)
		i=i-restx;
	if(blockIdx.y==gridy-1)
		j=j-resty;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Thread 在 Block 中的相对位置，和 i, j 有关
    int bid = blockIdx.y * gridDim.x + blockIdx.x; // Block 在 Grid 中的相对位置

    thread_diff[tid] = 0.0; // 初始化
 
    new_u[i * (Ny + 2) + j] = 0.25 * (
        u[(i - 1) * (Ny + 2) + j] + u[(i + 1) * (Ny + 2) + j] +
        u[i * (Ny + 2) + (j - 1)] + u[i * (Ny + 2) + (j + 1)]
    ); // Jacobi 迭代

    float diff = fabs(new_u[i * (Ny + 2) + j] - u[i * (Ny + 2) + j]); 
    thread_diff[tid] = diff;
    __syncthreads(); // 计算迭代 diff, 存储，所有线程同步

    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            thread_diff[tid] = fmax(thread_diff[tid], thread_diff[tid + s]);
        }
        __syncthreads();
    } 
    // Block 内所有 Threads 规约，最大值存存储在thread_diff[0] 
    //（没搞懂为什么是对的，GPT说是对的，结果也可以验证）

    if (tid == 0){
        blk_diff[bid] = thread_diff[0];
    } // Block 内最大 diff 存入 blk_diff 的对应位置
}


// 初始化 CUDA 设备，拷贝数据到 GPU 并执行内核
extern "C" void cuda_jacobi(float *u, float *new_u, int local_Nx, int local_Ny, float *local_diff) {

    // 查询设备信息
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0); 
    // printf("Global memory: %lu bytes\n", prop.totalGlobalMem);
    // printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    // printf("Warp size: %d\n", prop.warpSize);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 分配 Block 规模
	int gridx=local_Nx/BLOCK_SIZE;
	int gridy=local_Ny/BLOCK_SIZE;
	int restx=0, resty=0;
	if(local_Nx%BLOCK_SIZE!=0)
	{
		gridx++;
		restx=gridx*BLOCK_SIZE-local_Nx;
	}
	if(local_Ny%BLOCK_SIZE!=0)
	{
		gridy++;
		resty=gridy*BLOCK_SIZE-local_Ny;
	}
    //dim3 grid(local_Nx / BLOCK_SIZE, local_Ny / BLOCK_SIZE); // 分配 Grid 规模
	dim3 grid(gridx,gridy); // 分配 Grid 规模
 
    float *d_u, *d_new_u, *d_local_diff; //定义 Device 上的 u, new_u, local_diff

    size_t u_size = (local_Nx + 2) * (local_Ny + 2) * sizeof(float); // d_u, d_new_u 的内存空间
    //size_t l_diff_size = local_Nx / BLOCK_SIZE * local_Ny / BLOCK_SIZE * sizeof(float); // d_local_diff 的内存空间
	size_t l_diff_size=gridx*gridy*sizeof(float);

    cudaMalloc((void **)&d_u, u_size);
    cudaMalloc((void **)&d_new_u, u_size);
    cudaMalloc((void **)&d_local_diff, l_diff_size);// cudaMalloc 分配 Device 上的内存空间

    cudaMemcpy(d_u, u, u_size, cudaMemcpyHostToDevice); // 数组 u, Host 到 Device 初始化
    cudaMemcpy(d_new_u, new_u, u_size, cudaMemcpyHostToDevice); // 数组 new_u, Host 到 Device 初始化
    cudaMemset(d_local_diff, 0, l_diff_size); // 数组 d_local_diff, 初始化为0

    jacobi_update<<<grid, block>>>(d_u, d_new_u, local_Nx, local_Ny,gridx, gridy, restx, resty, d_local_diff); // Jacobi 迭代，执行 kernel

    cudaError_t err = cudaGetLastError(); // 确认kernel是否成功执行
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(new_u, d_new_u, u_size, cudaMemcpyDeviceToHost); // 数组 new_u, Device 到 Host 更新
    cudaMemcpy(local_diff, d_local_diff, l_diff_size, cudaMemcpyDeviceToHost); // 数组 local_diff, Device 到 Host 更新

    cudaFree(d_u);
    cudaFree(d_new_u);
    cudaFree(d_local_diff); // 释放 Device 内存空间
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define MAX_ITER 50000 // 最大迭代次数
#define MAX_REQUEST 5000 // 非阻塞通信寄存池上限
#define BLOCK_SIZE 32  // 和（共享内存变量 thread_diff，存储一个 Block 内 所有 Threads 的 diff）有关
#define TOL 1e-3       // 收敛容忍度

extern void cuda_jacobi(float *u, float *new_u, int local_Nx, int local_Ny, float *local_diff); // Jacobi迭代, 更新
extern void cuda_gauss_seidel(float *u, float *new_u, int local_Nx, int local_Ny, float *local_diff) ; // Gauss-Seidal迭代, 更新
void initialize_u(float *u, int Nx, int Ny, int rank, int dims[2], int coords[2]); // 初始化 u 数组
void write_to_file(float *u, int local_Nx, int local_Ny, int dims[2], int coords[2], const int itype); // 并行IO, 数据存储在data.dat

int main(int argc, char *argv[]) { //示例 mpirun -np 4 ./laplace_solver 4096
    if (argc < 3) {
        printf("Usage: %s <u_dimension>\n", argv[0]);
        printf("Usage: %s <iter_type>, jacobi = 0, gauss-seidal = 1 \n", argv[0]);
        return 0;
    }
    const int N = atoi(argv[1]); // Laplace 离散网格尺度
    const int itype = atoi(argv[2]); //确定迭代方式
    
    int rank, size; // MPI 进程初始化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};  // 创建二维进程拓扑
    MPI_Dims_create(size, 2, dims); // 自动划分进程为2D网格
    int periods[2] = {0, 0};       // 非周期性边界
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2]; // MPI进程的相对位置
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // 邻居进程的rank
    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    // 每个进程的局部网格大小
    int local_Nx = N / dims[0];
    int local_Ny = N / dims[1];
	if(coords[0]==dims[0]-1)
	{
		local_Nx=N-local_Nx*(dims[0]-1);
	}
	if(coords[1]==dims[1]-1)
	{
		local_Ny=N-local_Ny*(dims[1]-1);
	}

    // 使用一维数组来分配内存
    float *u = (float *)malloc((local_Nx + 2) * (local_Ny + 2) * sizeof(float));
    float *new_u = (float *)malloc((local_Nx + 2) * (local_Ny + 2) * sizeof(float));
    
    initialize_u(u, local_Nx, local_Ny, rank, dims, coords);
    initialize_u(new_u, local_Nx, local_Ny, rank, dims, coords);//初始化 u 和 new_u
	

    int iter; // 迭代次数
    float global_max_diff; // 全局迭代误差 
    
    float start_time = MPI_Wtime(); // MPI 计时
	
	int gridx=local_Nx/BLOCK_SIZE;
	int gridy=local_Ny/BLOCK_SIZE;
	if(local_Nx%BLOCK_SIZE!=0)
	{
		gridx++;
	}
	if(local_Ny%BLOCK_SIZE!=0)
	{
		gridy++;
	}
	
	//printf("%d and %d, %d and %d from %d\n",local_Nx, local_Ny, gridx,gridy,rank);

    for (iter = 0; iter < MAX_ITER; iter++) {
        // if(rank == 0 && iter % 10 == 0) printf("iter = %d\n",iter); // 显示迭代次数，当进度条用

        // 边界通信（非阻塞）
        //MPI_Request requests[MAX_REQUEST];
        //int request_count = 0;

        // 上下方向通信
        if (coords[0] != 0) {
            //MPI_Irecv(&u[0 * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, up, 1000*(up+1), cart_comm, &requests[request_count++]); // 接收上方
            //MPI_Isend(&u[1 * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, up, 100*(rank+1), cart_comm, &requests[request_count++]);  // 发送到下方
			MPI_Sendrecv(&u[1 * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, up, 100*(rank+1),&u[0 * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, up, 1000*(up+1), cart_comm, MPI_STATUS_IGNORE);
			//printf("%d send %d to %d and receives %d from %d\n",rank,local_Ny+2,up,local_Ny+2,up);
        } 
        else {
            for (int j = 0; j < local_Ny + 2; j++)  u[0 * (local_Ny + 2) + j] = 0.0; // MPI进程在求解域边界上，上边界条件
        }
		//printf("pass 1 from %d\n",rank);

        if (coords[0] != dims[0] - 1) {
            //MPI_Irecv(&u[(local_Nx + 1) * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, down, 100*(down+1), cart_comm, &requests[request_count++]); // 接收下方
            //MPI_Isend(&u[local_Nx * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, down, 1000*(rank+1), cart_comm, &requests[request_count++]); // 发送到上方
			MPI_Sendrecv(&u[local_Nx * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, down, 1000*(rank+1),&u[(local_Nx + 1) * (local_Ny + 2)], local_Ny + 2, MPI_FLOAT, down, 100*(down+1), cart_comm, MPI_STATUS_IGNORE);
			//printf("%d send %d to %d and receives %d from %d\n",rank,local_Ny+2,down,local_Ny+2,down);
        } 
        else {
            for (int j = 0; j < local_Ny + 2; j++)  u[(local_Nx + 1) * (local_Ny + 2) + j] = 1.0; // MPI进程在求解域边界上，下边界条件
        }
		//printf("pass 2 from %d\n",rank);
		//MPI_Barrier(cart_comm);
        // 左右方向通信
        if (coords[1] != 0) {
            for (int i = 1; i < local_Nx + 1; i++) {
                //MPI_Irecv(&u[i * (local_Ny + 2)], 1, MPI_FLOAT, left, 100000*(left+1)+i, cart_comm, &requests[request_count++]);   // 接收左侧
                //MPI_Isend(&u[i * (local_Ny + 2) + 1], 1, MPI_FLOAT, left, 10000*(rank+1)+i, cart_comm, &requests[request_count++]);   // 发送到左侧，tag为1
				MPI_Sendrecv(&u[i * (local_Ny + 2) + 1], 1, MPI_FLOAT, left, 10000*(rank+1)+i,&u[i * (local_Ny + 2)], 1, MPI_FLOAT, left, 100000*(left+1)+i, cart_comm, MPI_STATUS_IGNORE); 
			}
			
        } 
        else {
            for (int j = 0; j < local_Ny + 2; j++)  u[j * (local_Ny + 2) ] = 1.0; // MPI进程在求解域边界上，左边界条件
        }
		//printf("pass 3 from %d\n",rank);

        if (coords[1] != dims[1] - 1) {
            for (int i = 1; i < local_Nx + 1; i++) {
                //MPI_Irecv(&u[i * (local_Ny + 2) + local_Ny + 1], 1, MPI_FLOAT, right, 10000*(right+1)+i, cart_comm, &requests[request_count++]); // 接收右侧，tag为1
                //MPI_Isend(&u[i * (local_Ny + 2) + local_Ny], 1, MPI_FLOAT, right, 100000*(rank+1)+i, cart_comm, &requests[request_count++]);     // 发送到右侧
				MPI_Sendrecv(&u[i * (local_Ny + 2) + local_Ny], 1, MPI_FLOAT, right, 100000*(rank+1)+i,&u[i * (local_Ny + 2) + local_Ny + 1], 1, MPI_FLOAT, right, 10000*(right+1)+i, cart_comm, MPI_STATUS_IGNORE);
			
			}
			
        } 
        else {
            for (int j = 0; j < local_Ny + 2; j++)  u[j * (local_Ny + 2) + local_Ny + 1] = 1.0; // MPI进程在求解域边界上，右边界条件
        }
		//printf("pass 4 from %d\n",rank);
		//printf("where from %d\n",rank);
		//printf("request %d from %d\n",request_count,rank);
        //MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE); // MPI 进程同步
		MPI_Barrier(cart_comm);
		//printf("%d:here from %d\n",iter,rank);
		
		
        //int l_diff_size = local_Nx  / BLOCK_SIZE * local_Ny / BLOCK_SIZE ; // 数组 local_diff 维度
		int l_diff_size = gridx*gridy;

        float* local_diff = (float *)malloc(l_diff_size * sizeof(float)); // 数组 local_diff, 存储 CUDA kernel 中每个 Block 的最大diff
        for (int i = 0; i < l_diff_size; ++i)  local_diff[i] = 0.0; // 初始化 local_diff

        // 注释掉的这一部分可以查看每次Kernel执行的时间
        // cudaEvent_t start, stop;
        // float elapsedTime;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start,0);

        // Jacobi更新
		//printf("here2 from %d\n",rank);
        if(itype == 0)
            cuda_jacobi(u, new_u, local_Nx, local_Ny, local_diff);

        // Gauss-Seidal更新
        if(itype == 1)
            cuda_gauss_seidel(u, new_u, local_Nx, local_Ny, local_diff);

        // cudaEventRecord(stop,0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&elapsedTime, start, stop);
        // printf("Kernel execution time: %f ms\n", elapsedTime);

        // 交换指针
        float *temp = u;
        u = new_u;
        new_u = temp;

        for (int i = l_diff_size - 1 ; i > 0; i--) {
            local_diff[i - 1] = fmax(local_diff[i] , local_diff[i - 1]) ;
        } // 找到 local_diff 最大值， 存入 local_diff[0]

        float max_diff = local_diff[0];
		MPI_Barrier(cart_comm);
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_FLOAT, MPI_MAX, cart_comm);
        // 收敛检查, 所有MPI进程规约， global_max_diff为本次迭代最大误差
		//printf("here3 from %d, %f\n",rank,global_max_diff);

        if(global_max_diff < TOL)
			break; // 满足条件，跳出迭代
		
    }

    float end_time = MPI_Wtime();
    if (rank == 0) {    
        printf("Iteration count: %d\n", iter);
        printf("Final tolerance: %.8f\n", global_max_diff);
        printf("Elapsed time: %.8f seconds\n", end_time - start_time);
    }// 打印迭代次数，最终误差，MPI+CUDA并行部分总耗时

    write_to_file(u, local_Nx, local_Ny, dims, coords, itype);

    free(u);
    free(new_u);
    MPI_Finalize();
    return 0;
}

// Dirichlet边界条件，初始化 u 数组
void initialize_u(float *u, int Nx, int Ny, int rank, int dims[2], int coords[2]) {
    for (int i = 0; i < Nx + 2; i++) {
        for (int j = 0; j < Ny + 2; j++) {
            u[i * (Ny + 2) + j] = 0.0;  // 初始化所有值为0
        }
    }
    if (coords[0] == 0) { // 顶边界 = 0
        for (int j = 0; j < Ny + 2; j++) {
            u[0 * (Ny + 2) + j] = 0.0;
        }
    }
    if (coords[0] == dims[0] - 1) { // 底边界 = 1
        for (int j = 0; j < Ny + 2; j++) {
            u[(Nx + 1) * (Ny + 2) + j] = 1.0;
        }
    }
    if (coords[1] == 0) { // 左边界 = 1
        for (int i = 0; i < Nx + 2; i++) {
            u[i * (Ny + 2) ] = 1.0;
        }
    }
    if (coords[1] == dims[1] - 1) { // 右边界 = 1
        for (int i = 0; i < Nx + 2; i++) {
            u[i * (Ny + 2) + Ny + 1] = 1.0;
        }
    }
}
void write_to_file(float *u, int local_Nx, int local_Ny, int dims[2], int coords[2], const int itype) { // 并行IO, 数据存储在data.dat
    MPI_File file;
    MPI_Status status;
    if(itype == 0)
        MPI_File_open(MPI_COMM_WORLD, "u_jacobi.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    if(itype == 1)
        MPI_File_open(MPI_COMM_WORLD, "u_gauss_seidal.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    // 分行输出
    for (int i = 1; i <= local_Nx; i++) {
        // 根据进程位置和行数计算偏移量
        MPI_Offset offset = (coords[0] * local_Nx * local_Ny * dims[1]) + (i - 1) * local_Ny * dims[1] + coords[1] * local_Ny;
        MPI_File_write_at(file, offset * sizeof(float), &u[i * (local_Ny + 2) + 1], local_Ny , MPI_FLOAT, &status);
    }
    // 确保所有进程都完成写入
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&file);
}


MPICC = mpicc
NVCC = nvcc
TARGET = laplace_solver

# 设置CUDA相关的编译和链接标志
CUDA_FLAGS = -arch=sm_89 # 测试显卡为 Nvidia Tesla P4, 更新架构的显卡可以使用更高的参数
CUDA_LIBS = -lcudart -lm
CFLAGS = -O3 -Wall
LDFLAGS = -L/usr/local/cuda-12.6/targets/x86_64-linux/lib # 测试版本为 CUDA-11.8, 更换测试平台时注意更换路径
INCLUDES = -I/usr/local/cuda-12.6/targets/x86_64-linux/include

all: $(TARGET)

$(TARGET): main.o cuda_jacobi.o cuda_gauss.o
	$(MPICC) main.o cuda_jacobi.o cuda_gauss.o -o $(TARGET) $(LDFLAGS) $(CUDA_LIBS)

main.o: main.c
	$(MPICC) $(CFLAGS) -c main.c -o main.o $(INCLUDES)

cuda_jacobi.o: cuda_jacobi.cu
	$(NVCC) -c cuda_jacobi.cu -o cuda_jacobi.o $(CUDA_FLAGS)

cuda_gauss.o: cuda_gauss.cu
	$(NVCC) -c cuda_gauss.cu -o cuda_gauss.o $(CUDA_FLAGS)

clean:
	rm -f *.o $(TARGET)

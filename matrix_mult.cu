#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MATRIX_SIZE 10000
#define RANGE_NUMBERS 10
#define DIM_THREADS 32

__host__ __device__ inline void setAt(int *m, int i, int j, int v) {
    *(m + i*MATRIX_SIZE + j) = v;
}

__host__ __device__ inline  int getAt(int *m, int i, int j) {
    return *(m + i*MATRIX_SIZE + j);
}

/*__global__ void matrix_mult_shared_mem(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int shared_a[MATRIX_SIZE*MATRIX_SIZE];
    __shared__ int shared_b[MATRIX_SIZE*MATRIX_SIZE];

    for(int aux = 0; aux < MATRIX_SIZE; aux++) {
        setAt(shared_a, i, aux, getAt(a, i, aux));
        setAt(shared_b, i, aux, getAt(b, i, aux));
    }
    __syncthreads();

    int sum = 0;
    for(int it = 0; it < MATRIX_SIZE; it++) {
        sum += (getAt(shared_a, i, it) * getAt(shared_b, it, j)) % 50;
    }
    setAt(c, i, j, sum);
}*/

__global__ void matrix_mult(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;
    for(int it = 0; it < MATRIX_SIZE; it++) {
        sum += (getAt(a, i, it) * getAt(b, it, j)) % 50;
    }
    setAt(c, i, j, sum);
}

int main(int argc, char **argv) {
    printf("Starting\n");
    srand(time(0));
    
    float etime;
    struct timespec t_start, t_end;
    
    size_t size = sizeof(int) * MATRIX_SIZE * MATRIX_SIZE;
    
    int * a, * b, * c;

    a = (int *) malloc(size);
    b = (int *) malloc(size);
    c = (int *) malloc(size);

    // fill matrices
    for(int i = 0; i < MATRIX_SIZE; i++) {
        for(int j = 0; j < MATRIX_SIZE; j++){
            setAt(a, i, j, rand() % RANGE_NUMBERS);
            setAt(b, i, j, rand() % RANGE_NUMBERS);
        }
    }

    int * d_a, * d_b, * d_c;

    // alloc memory in device
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    for(int dim_threads = DIM_THREADS; dim_threads >= 1; dim_threads >>= 1) {

        clock_gettime(CLOCK_REALTIME, &t_start);
        // copy matrices do device memory
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(dim_threads, dim_threads);
        dim3 numBlocks(MATRIX_SIZE / threadsPerBlock.x, MATRIX_SIZE / threadsPerBlock.y);

        // call function in device
        matrix_mult<<<numBlocks,threadsPerBlock>>>(d_a, d_b, d_c);

        // get data from device memory
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        clock_gettime(CLOCK_REALTIME, &t_end);

        etime = (t_end.tv_sec + t_end.tv_nsec / 1000000000.) -
                (t_start.tv_sec + t_start.tv_nsec / 1000000000.);

        printf("\nNum threads per block: %dx%d Time spent: %lf\n", dim_threads,dim_threads, etime);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n",cudaGetErrorString(err));
        }
        

    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

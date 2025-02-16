#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_THREADS_PER_BLOCK 256     // could be 1024 at most
#define SQRT_MAX_THREADS_PER_BLOCK 16 //

// stole this from StackOverflow to handle errors
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

typedef struct {
  float prep_time;
  float run_time;
  char *calling_fn;

} TestResult;

void printDim3(dim3 grid, dim3 block) {
  printf("Grid: (%d, %d, %d) blocks.\nBlocks: (%d, %d, %d) threads.\n", grid.x,
         grid.y, grid.z, block.x, block.y, block.z);
}

void printTestResult(TestResult tr) {
  printf("%s:\n\tPrep: %.2fus\n\tRun : %.2fus\n", tr.calling_fn, tr.prep_time,
         tr.run_time);
}

bool checkMatrix(int *arr, int n) {
  for (int i = 0; i < n * n; i++) {
    if (arr[i] != n) {
      printf("%d!-\n\n", arr[i]);
      return false;
    }
  }
  return true;
}

void printMatrix(int *arr, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%5d", arr[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

dim3 calcGridSize(int n, dim3 block) {
  int gridDimX = (n + block.x - 1) / block.x;
  int gridDimY = (n + block.y - 1) / block.y;
  return dim3(gridDimX, gridDimY, 1);
}

dim3 calcBlockSize(int n) {
  int blockDimX =
      (n < SQRT_MAX_THREADS_PER_BLOCK) ? n : SQRT_MAX_THREADS_PER_BLOCK;
  int blockDimY =
      (n < SQRT_MAX_THREADS_PER_BLOCK) ? n : SQRT_MAX_THREADS_PER_BLOCK;
  return dim3(blockDimX, blockDimY, 1);
}

__global__ void transposeKernel(int n, int *input, int *output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    output[col * n + row] = input[row * n + col];
  }
}

__global__ void coalescedOneDimKernel(int n, int *a, int *b, int *c) {
  const int x = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
  const int y = blockIdx.y * blockDim.y + (threadIdx.x % blockDim.y);

  if (x == 0 && y == 0) {
    printf("blockDim.x = %d, blockDim.y = %d\n", blockDim.x, blockDim.y);
    printf("(%d,%d), ", x, y);
  }
  // printf("(%d,%d), ", x, y);

  if (x < n && y < n) {
    int sum = 0;
    for (int k = 0; k < n; k++) {
      sum += a[x * n + k] * b[k * n + y];
    }
    c[x * n + y] = sum;
    // if (sum != n) printf("Error calculating from coalesced\tn = %d sum =
    // %d\n", n, sum);
  }
}

TestResult coalescedOneDim(int n) {
  cudaEvent_t prep, start, end;
  cudaEventCreate(&prep);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float ms_prep, ms_run;

  cudaEventRecord(prep);

  size_t capacity = n * n * sizeof(int);

  int *a, *b, *c;
  a = (int *)malloc(capacity);
  b = (int *)malloc(capacity);
  c = (int *)malloc(capacity);

  for (int i = 0; i < n * n; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  int *dev_a, *dev_b, *dev_c;
  cudaMalloc(&dev_a, capacity);
  cudaMalloc(&dev_b, capacity);
  cudaMalloc(&dev_c, capacity);

  gpuErrchk(cudaMemcpy(dev_a, a, capacity, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dev_b, b, capacity, cudaMemcpyHostToDevice));

  dim3 dimBlock = calcBlockSize(n);
  dim3 dimGrid = calcGridSize(n, dimBlock);
  printDim3(dimGrid, dimBlock);

  cudaEventRecord(start);

  coalescedOneDimKernel<<<dimGrid, dimBlock>>>(n, dev_a, dev_b, dev_c);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(c, dev_c, capacity, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&ms_run, start, end);
  cudaEventElapsedTime(&ms_prep, prep, start);

  TestResult tr;
  tr.calling_fn = "coalescedOneDim";
  tr.prep_time = ms_prep * 1000;
  tr.run_time = ms_run * 1000;

  free(a);
  free(b);
  free(c);

  return tr;
}

__global__ void transOneDimKernel(int n, int *a, int *b_t, int *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    int sum = 0;
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b_t[col * n + k];
    }
    c[row * n + col] = sum;
  }
}

TestResult transOneDim(int n) {
  cudaEvent_t prep, start, end;
  cudaEventCreate(&prep);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float ms_prep, ms_run;

  cudaEventRecord(prep);

  size_t capacity = n * n * sizeof(int);

  int *a, *b, *c;
  a = (int *)malloc(capacity);
  b = (int *)malloc(capacity);
  c = (int *)malloc(capacity);

  for (int i = 0; i < n * n; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  int *dev_a, *dev_b, *dev_b_t, *dev_c;
  cudaMalloc(&dev_a, capacity);
  cudaMalloc(&dev_b, capacity);
  cudaMalloc(&dev_b_t, capacity);
  cudaMalloc(&dev_c, capacity);

  gpuErrchk(cudaMemcpy(dev_a, a, capacity, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dev_b, b, capacity, cudaMemcpyHostToDevice));
  // gpuErrchk(cudaMemcpy(dev_c, c, capacity, cudaMemcpyHostToDevice));

  dim3 dimBlock = calcBlockSize(n);
  dim3 dimGrid = calcGridSize(n, dimBlock);

  cudaEventRecord(start);

  transposeKernel<<<dimGrid, dimBlock>>>(n, dev_b, dev_b_t);
  gpuErrchk(cudaPeekAtLastError());

  transOneDimKernel<<<dimGrid, dimBlock>>>(n, dev_a, dev_b_t, dev_c);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(c, dev_c, capacity, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_b_t);
  cudaFree(dev_c);

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&ms_run, start, end);
  cudaEventElapsedTime(&ms_prep, prep, start);

  TestResult tr;
  tr.calling_fn = "transOneDim";
  tr.prep_time = ms_prep * 1000;
  tr.run_time = ms_run * 1000;

  free(a);
  free(b);
  free(c);

  return tr;
}

__global__ void naiveOneDimKernel(int n, int *a, int *b, int *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    int sum = 0;
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
  }
}

TestResult naiveOneDim(int n) {
  cudaEvent_t prep, start, end;
  cudaEventCreate(&prep);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float ms_prep, ms_run;

  cudaEventRecord(prep);

  size_t capacity = n * n * sizeof(int);

  int *a, *b, *c;
  a = (int *)malloc(capacity);
  b = (int *)malloc(capacity);
  c = (int *)malloc(capacity);

  for (int i = 0; i < n * n; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  int *dev_a, *dev_b, *dev_c;
  cudaMalloc(&dev_a, capacity);
  cudaMalloc(&dev_b, capacity);
  cudaMalloc(&dev_c, capacity);

  gpuErrchk(cudaMemcpy(dev_a, a, capacity, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(dev_b, b, capacity, cudaMemcpyHostToDevice));
  // gpuErrchk(cudaMemcpy(dev_c, c, capacity, cudaMemcpyHostToDevice));

  dim3 dimBlock(calcBlockSize(n));
  dim3 dimGrid(calcGridSize(n, dimBlock));

  cudaEventRecord(start);
  naiveOneDimKernel<<<dimGrid, dimBlock>>>(n, dev_a, dev_b,
                                           dev_c); // block, threads per block
  gpuErrchk(cudaPeekAtLastError());

  // cudaMemcpy(a, dev_a, capacity, cudaMemcpyDeviceToHost);
  // cudaMemcpy(b, dev_b, capacity, cudaMemcpyDeviceToHost);
  cudaMemcpy(c, dev_c, capacity, cudaMemcpyDeviceToHost);

  if (!checkMatrix(c, n)) {
    fprintf(stderr, "ERROR: MATRIX MULTIPLICATION DIDN'T WORK!!!!!\n");
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&ms_run, start, end);
  cudaEventElapsedTime(&ms_prep, prep, start);

  free(a);
  free(b);
  free(c);

  TestResult tr;
  tr.calling_fn = "naiveOneDim";
  tr.prep_time = ms_prep * 1000;
  tr.run_time = ms_run * 1000;

  return tr;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: ./a.out N, where N is the matrix size expressed as "
                    "2 ** N. Exiting\n");
    return EXIT_FAILURE;
  }

  int n_exp = atoi(argv[1]);
  int n = 1 << n_exp;

  TestResult naive_one_dim = naiveOneDim(n);
  printTestResult(naive_one_dim);

  TestResult transOneDim_tr = transOneDim(n);
  printTestResult(transOneDim_tr);

  TestResult coalesced_one_dim = coalescedOneDim(n);
  printTestResult(coalesced_one_dim);

  return EXIT_SUCCESS;
}

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void stone_scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  // scan method in class
  __shared__ float XY[BLOCK_SIZE];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len) XY[threadIdx.x] = input[i];
  __syncthreads();
  float temp = XY[0];
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride)
      temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
    __syncthreads();
    XY[threadIdx.x] = temp; 
  }
  if (i < len){
    output[i] = XY[threadIdx.x];
  }
}

__global__ void kung_scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  // scan method in class
  // float T[2*BLOCK_SIZE] is in shared memory
  // forward scan
  __shared__ float T[2*BLOCK_SIZE];
  int i = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int j = i + 1;
  if (i < len) T[2*threadIdx.x] = input[i];
  if (j < len) T[2*threadIdx.x+1] = input[j];
  __syncthreads();
  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];  
    stride *= 2;
  }
  // post scan
  stride = 2*BLOCK_SIZE/4;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE) 
      T[index+stride] += T[index];
    stride /= 2; 
  }
  __syncthreads();
  if (i < len){
    output[i] = T[2*threadIdx.x];
  }
  if (j < len){
    output[j] = T[2*threadIdx.x+1];
  }
}

__global__ void stone_gen_Last_Item(float *output, float *last_list, int num_blocks) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // get those last items into the list;
  if ((threadIdx.x == blockDim.x - 1) && (blockIdx.x != (num_blocks-1))){
    last_list[blockIdx.x] = output[i];
  }
}

__global__ void kung_gen_Last_Item(float *output, float *last_list, int num_blocks) {
  int i = 2*(blockIdx.x * blockDim.x + threadIdx.x)+1;
  if ((threadIdx.x == blockDim.x - 1) && (blockIdx.x != num_blocks-1)){
    last_list[blockIdx.x] = output[i];
  }
}

__global__ void stone_block_addition(float* output, float* last_list, int num_blocks, int len){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float temp = output[i];
  for (unsigned int j = 0; j < num_blocks-1; j++) {
    if (blockIdx.x > j) {
      temp += last_list[j];
    }else{
      break;
    }
  }
  if (i < len){
    output[i] = temp;
  }
}

__global__ void kung_block_addition(float* output, float* last_list, int num_blocks, int len){
  int i = 2*(blockIdx.x * blockDim.x + threadIdx.x);
  int j = i + 1;
  float temp1 = output[i];
  float temp2 = output[j];
  for (unsigned int k = 0; k < num_blocks-1; k++) {
    if (blockIdx.x > k) {
      temp1 += last_list[k];
      temp2 += last_list[k];
    }else{
      break;
    }
  }
  if (i < len){
    output[i] = temp1;
  }
  if (j < len){
    output[j] = temp2;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *last_list;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  int num_blocks = numElements/(2*BLOCK_SIZE) + 1;
  // int num_blocks = numElements/(BLOCK_SIZE) + 1;
  cudaMalloc((void **)&last_list, num_blocks * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(num_blocks,1,1);
  dim3 dimBlock(BLOCK_SIZE,1,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  kung_scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
  kung_gen_Last_Item<<<dimGrid, dimBlock>>>(deviceOutput, last_list, num_blocks);
  kung_block_addition<<<dimGrid, dimBlock>>>(deviceOutput, last_list, num_blocks, numElements);
  //@@ on the deivce

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(last_list);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);
  // for (int i = 0; i < numElements; i++)
  //   cout << hostOutput[i] << endl;

  free(hostInput);
  free(hostOutput);

  return 0;
}

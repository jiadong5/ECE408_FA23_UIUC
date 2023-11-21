
#include <wb.h>
#include <iostream>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_WIDTH 16
#define TILE_WIDTH 16


// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  // define tiles
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  // load data to tiles
  for (int q = 0; q < (ceil((float)numAColumns/TILE_WIDTH)); ++q) { 
    if (Row < numARows && (q*TILE_WIDTH+tx) < numAColumns)
      subTileM[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH + tx];
    else
      subTileM[ty][tx] = 0;
    if (Col < numBColumns && (q*TILE_WIDTH+ty) < numBRows)
      subTileN[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns + Col];
    else
      subTileN[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++){
      Pvalue += subTileM[ty][k]*subTileN[k][tx];
    }  
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns){
    C[Row*numCColumns + Col] = Pvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCColumns*numCRows*sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceA, numAColumns*numARows*sizeof(float));
  cudaMalloc((void**) &deviceB, numBColumns*numBRows*sizeof(float));
  cudaMalloc((void**) &deviceC, numCColumns*numCRows*sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  // cudaMemcpy is the instruction that communicates device and host.
  // (Destination, Source, Size, Type)
  cudaMemcpy(deviceA, hostA, numAColumns*numARows*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBColumns*numBRows*sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numCColumns + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
               (numCRows + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 1);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                        numARows, numAColumns,
                                        numBRows, numBColumns,
                                        numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  // (Destination, Source, Size, Type)
  cudaMemcpy(hostC, deviceC, numCColumns*numCRows*sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < numCRows; i++){
  //     for (int j = 0; j < numCColumns; ++j) {
  //         std::cout << hostC[i * numCColumns + j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

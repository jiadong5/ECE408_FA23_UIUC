// Histogram Equalization

#include <wb.h>
#include <iostream>

#define HISTOGRAM_LENGTH 256
#define BLOCK_WIDTH 128
#define TILE_WIDTH 16


//@@ insert code here
// float image to unsigned char image.
__global__ void float2uchar(int imageWidth, int imageHeight, int imageChannels, 
                            unsigned char* ucharImage, float* inputImage){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Channel = threadIdx.z;
  if (Col < imageWidth && Row < imageHeight){
    int index = (Row * imageWidth + Col) * imageChannels +Channel;
    ucharImage[index] = (unsigned char) (255*inputImage[index]); 
  }
}

// rgb image to gray scale
// 2D kernel
__global__ void RGB2GrayScale(int imageWidth, int imageHeight, int imageChannels, 
                              unsigned char* ucharImage, unsigned char* grayImage){
  int Col = blockDim.x*blockIdx.x + threadIdx.x;
  int Row = blockDim.y*blockIdx.y + threadIdx.y;
  if (Row < imageHeight && Col < imageWidth){
    int index = Row * imageWidth + Col;
    unsigned char r = ucharImage[imageChannels*index];
    unsigned char g = ucharImage[imageChannels*index+1];
    unsigned char b = ucharImage[imageChannels*index+2];
    grayImage[index] = (unsigned char) (0.21*r +0.71*g+0.07*b);
  }
}

// Histogramming kernel.
// using 1D kernel
__global__ void ImageHisto(int imageWidth, int imageHeight, unsigned char* grayImage, unsigned int* Histo) {
  // initialize shared memory.
  // block size = HISTOGRAM_LENGTH.
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  int index = blockDim.x * threadIdx.y + threadIdx.x;
  if (index < HISTOGRAM_LENGTH)
    histo_private[index] = 0;

  __syncthreads();

  int Col = blockDim.x*blockIdx.x + threadIdx.x;
  int Row = blockDim.y*blockIdx.y + threadIdx.y;

  if (Row < imageHeight && Col < imageWidth) {
    int pixel_index = Row * imageWidth + Col;
    atomicAdd(&(histo_private[grayImage[pixel_index]]), 1);
  }

  __syncthreads();
  // write to global memory.
  if (index < HISTOGRAM_LENGTH)
    atomicAdd( &(Histo[index]), histo_private[index]);
}

// Just slightly modify the code.
__global__ void ScanCDF(float *cdf, unsigned int* histo, int imgSize){
    __shared__ float T[HISTOGRAM_LENGTH];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int id1 = start + 2 * t;
    unsigned int id2 = start + 2 * t + 1;
    int stride = 1;
    int index;

    // initialize shared memory
    T[2 * t] = (float) histo[id1] / imgSize;
    T[2 * t + 1] = (float) histo[id2] / imgSize;

    // Reduction Step
    while (stride < HISTOGRAM_LENGTH) {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_LENGTH && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride = stride * 2;
    }

    // Post Scan Step
    stride = HISTOGRAM_LENGTH / 4;
    while (stride > 0) {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if ((index + stride) < HISTOGRAM_LENGTH)
            T[index + stride] += T[index];
        stride = stride / 2;
    }

    __syncthreads();

    // copy back to global memory
    cdf[id1] = T[2 * t];
    cdf[id2] = T[2 * t + 1];
}

// correct_color。
__global__ void CorrectColor(unsigned char* ucharImage, float* cdf, int imageWidth, int imageHeight, int imageChannels){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Channel = threadIdx.z;

  if (Col < imageWidth && Row < imageHeight){
    int index = (Row * imageWidth + Col) * imageChannels + Channel;
    ucharImage[index] = min(max(255.0f*(cdf[ucharImage[index]] - cdf[0])/(1.0f - cdf[0]), 0.0f), 255.0f);
  }
}

// uchar to float kernel.
__global__ void uchar2float(int imageWidth, int imageHeight, int imageChannels, 
                            unsigned char* ucharImage, float* outputImage){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Channel = threadIdx.z;

  if (Col < imageWidth && Row < imageHeight){
    int index = (Row * imageWidth + Col) * imageChannels +Channel;
    outputImage[index] = ((float) ucharImage[index])/255.0f;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceUcharImage; 
  unsigned char *deviceGrayImage;
  unsigned int *deviceHisto;
  float* deviceCDF;
  float* deviceOutputImage;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  // allocate cuda memory and copy data.
  int num_total = imageWidth * imageHeight * imageChannels;
  int pixel_total = imageWidth * imageHeight;
  cudaMalloc((void **)&deviceInputImage, num_total * sizeof(float));
  cudaMalloc((void **)&deviceUcharImage, num_total * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayImage, pixel_total * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceOutputImage, num_total * sizeof(float));

  cudaMemcpy(deviceInputImage, hostInputImageData, num_total * sizeof(float), cudaMemcpyHostToDevice);

  // SETUP KERNEL DIMENSIONS
  dim3 GridDimConvertion(ceil(((float) imageWidth)/TILE_WIDTH), ceil(((float) imageHeight)/TILE_WIDTH), 1);
  dim3 BlockDimConvertion(TILE_WIDTH, TILE_WIDTH, imageChannels);
  dim3 GridScan(1,1,1);
  dim3 BlockScan(BLOCK_WIDTH,1,1);
  dim3 GridDim2D(ceil(((float) imageWidth)/TILE_WIDTH), ceil(((float) imageHeight)/TILE_WIDTH), 1);
  dim3 BlockDim2D(TILE_WIDTH, TILE_WIDTH, 1);

  // LAUNCH KERNELS
  float2uchar<<<GridDimConvertion, BlockDimConvertion>>>(imageWidth, imageHeight, imageChannels, deviceUcharImage, deviceInputImage);
  RGB2GrayScale<<<GridDim2D, BlockDim2D>>>(imageWidth, imageHeight, imageChannels, deviceUcharImage, deviceGrayImage);
  ImageHisto<<<GridDim2D, BlockDim2D>>>(imageWidth, imageHeight, deviceGrayImage, deviceHisto);
  ScanCDF<<<GridScan, BlockScan>>>(deviceCDF, deviceHisto, pixel_total);
  CorrectColor<<<GridDimConvertion, BlockDimConvertion>>>(deviceUcharImage, deviceCDF, imageWidth, imageHeight, imageChannels);
  uchar2float<<<GridDimConvertion, BlockDimConvertion>>>(imageWidth, imageHeight, imageChannels, deviceUcharImage, deviceOutputImage);

  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutputImage, num_total * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(deviceInputImage);
  cudaFree(deviceUcharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);
  cudaFree(deviceOutputImage);
  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}

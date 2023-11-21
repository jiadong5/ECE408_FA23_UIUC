#include "cpu-new-forward.h"

void conv_forward_cpu(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
  /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)

    Function paramters:
    output - output
    input - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

  const int H_out = (H - K)/S + 1;
  const int W_out = (W - K)/S + 1;
  

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  // An example use of these macros:
  // float a = in_4d(0,0,0,0)
  // out_4d(0,0,0,0) = a
  #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your CPU convolution kernel code here
  // for b = 0 .. Batch                     // for each image in the batch 
  //   for m = 0 .. Map_out               // for each output feature maps
  //       for h = 0 .. Height_out        // for each output element
  //           for w = 0 .. Width_out 
  //           {
  //               output[b][m][h][w] = 0;
  //               for c = 0 .. Channel   // sum over all input feature maps
  //                   for p = 0 .. K // KxK filter
  //                       for q = 0 .. K
  //                           output[b][m][h][w] += input[b][c][h * Stride + p][w * Stride + q] * k[m][c][p][q]
  //           }
  for (int b = 0; b < B; b++){
    for (int m = 0; m < M; m++){
      for (int h = 0; h < H_out; h++){
        for (int w = 0; w < W_out; w++){
          out_4d(b,m,h,w) = 0;
          for (int c = 0; c < C; c++){
            for (int p = 0; p < K; p++){
              for (int q = 0; q < K; q++){
                out_4d(b,m,h,w) += in_4d(b,c,(h*S+p),(w*S+q))*mask_4d(m,c,p,q);
              }
            }
          }
        }
      }
    }
  }
  
  #undef out_4d
  #undef in_4d
  #undef mask_4d
  return;
}
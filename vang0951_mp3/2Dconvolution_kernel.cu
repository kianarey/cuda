#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{
  __shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row_o = blockIdx.y * TILE_SIZE + ty;
  int col_o = blockIdx.x * TILE_SIZE + tx;

  int row_i = row_o - (KERNEL_SIZE/2);
  int col_i = col_o - (KERNEL_SIZE/2);

  float P_val = 0.0f;

  if((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width)){
    N_s[ty][tx] = N.elements[row_i * N.width + col_i];
  }
  else{
    N_s[ty][tx] = 0.0;
  }
  __syncthreads();
    
  if(ty < TILE_SIZE && tx < TILE_SIZE ){
    for (int i = 0; i < KERNEL_SIZE; i++){
      for (int j = 0; j < KERNEL_SIZE; j++){
	P_val += Mc[i * KERNEL_SIZE + j] * N_s[i + ty][j + tx];
      }
    }
    __syncthreads();
  
    if(row_o < P.height && col_o < P.width){
      P.elements[row_o * P.width + col_o] = P_val;
    }
  }
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_

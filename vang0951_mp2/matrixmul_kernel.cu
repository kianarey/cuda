/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#include <math.h>


// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  __shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
  __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Identify the row and column of P_d element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float P_val = 0.0;

  // Loop over the M_ and N_ tiles to compute the P_d element
  
  for (int m = 0; m < ceilf(M.width/(float)TILE_WIDTH); ++m){
    int M_row = Row;
    int M_col = m * TILE_WIDTH + tx; // from lecture 5

    int N_row = m * TILE_WIDTH + ty;
    int N_col = Col;
    
    // Loading M
    if((M_row < M.height) && (M_col < M.width)){
      M_s[ty][tx] = M.elements[M_row * M.width + M_col];
    }
    else{
      M_s[ty][tx] = 0.0;
    }
      
    // Loading N
    if((N_row < N.height) && (N_col < N.width)){
      N_s[ty][tx] = N.elements[N_row * N.width + N_col];
    }
    else{
      N_s[ty][tx] = 0.0;
    }
    __syncthreads();
      
    
    for (int k = 0; k < TILE_WIDTH; ++k){
      P_val += M_s[ty][k] * N_s[k][tx]; // Perform summation of dot product
    }
    __syncthreads();
  }
  
  if (Row < P.height && Col < P.width){
    P.elements[Row * P.width + Col] = P_val;
  }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_

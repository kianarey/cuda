/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  float P_val = 0.0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  if((row < MATRIX_SIZE) && (col < MATRIX_SIZE)){ // check bounds
    for(int i = 0; i < MATRIX_SIZE; i++){ // CUDA C uses row major layout
      float M_element = M.elements[row * M.width + i]; // access element row by row

      float N_element = N.elements[i * N.width + col]; // access element col by col
      P_val += M_element * N_element; //perform inner product
    }

    //Store P_val in P where each thread writes an element
    P.elements[row * P.width + col] = P_val; // the result P_val is the dot product of M_element (row) and N_element (col)
  }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_

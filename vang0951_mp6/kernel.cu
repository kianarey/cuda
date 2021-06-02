#include <stdio.h>
#include <math.h>
#include <iostream>

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr,
    unsigned int *csrColIdx, float *csrData, float *inVector,
    float *outVector) {

    // INSERT KERNEL CODE HERE
    // Code below is from Lecture 16 Sparse Methods, slide 12
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < dim){
      float dot = 0;
      int row_start = csrRowPtr[row];
      int row_end = csrRowPtr[row + 1];
      for (int i = row_start; i < row_end; i++){
        dot += csrData[i] * inVector[csrColIdx[i]];
      }
      outVector[row] += dot;
    }

}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm,
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx,
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

    // INSERT KERNEL CODE HERE
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < dim){
      float dot = 0;
      int row_start = 0;
      int row_end = jdsRowNNZ[row];
      for (int i = row_start; i < row_end; i++){
        dot += jdsData[jdsColStartIdx[i] + row] * inVector[jdsColIdx[jdsColStartIdx[i] + row]];
      }
      outVector[jdsRowPerm[row]] += dot;
    }

}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx,
    float *csrData, float *inVector, float *outVector) {

    // INSERT CODE HERE
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(dim/512.0), 1 ,1);
    spmv_csr_kernel<<<dimGrid, dimBlock>>>(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector);

}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ,
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData,
    float* inVector, float *outVector) {

    // INSERT CODE HERE
    dim3 dimBlock(512, 1, 1);
    dim3 dimGrid(ceil(dim/512.0), 1 ,1);
    spmv_jds_kernel<<<dimGrid, dimBlock>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx, jdsColIdx, jdsData, inVector, outVector);

}

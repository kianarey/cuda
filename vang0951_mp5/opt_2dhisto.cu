#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "opt_2dhisto.h"
#include "ref_2dhisto.h"


__global__ void histo_kernel(uint32_t* d_input, size_t inputSize, unsigned int* histo);

void opt_2dhisto(uint32_t* d_input, size_t inputSize, unsigned int* histo)
{
    /* This function should only contain grid setup
       code and a call to the GPU histogramming kernel.
       Any memory allocations and transfers must be done
       outside this function */
  cudaMemset(histo, 0, HISTO_WIDTH * HISTO_HEIGHT * sizeof(unsigned int));
  cudaDeviceProp dev_prop;
  cudaGetDeviceProperties(&dev_prop, 0); // check the first [and only] device

  //int maxThreadsPerMP = dev_prop.maxThreadsPerMultiProcessor;
  //int totalMP = dev_prop.multiProcessorCount;
  int maxThreadsPerBlock = dev_prop.maxThreadsPerBlock;

  //float maxBlocks = (maxThreadsPerMP * totalMP)/(float) maxThreadsPerBlock;

  dim3 dimGrid(ceil(inputSize/(float)maxThreadsPerBlock));
  dim3 dimBlock(maxThreadsPerBlock);

  // launch histo_kernel
  histo_kernel<<<dimGrid, dimBlock>>>(d_input, inputSize, histo);
  cudaDeviceSynchronize();
}
/* Include below the implementation of any other functions you need */

__device__ unsigned int atomicAddWithSaturation(unsigned int* address, unsigned int val, int sat_value){
  // Code from CUDA C++ programming

  unsigned int old = *address;
  unsigned int assumed;

    do{
      assumed = old;
      unsigned int temp = (val + assumed) >= sat_value ? sat_value : val + assumed;
      old = atomicCAS(address, assumed, temp);
    } while (assumed != old);
  return old;
}

__global__ void histo_kernel(uint32_t* d_input, size_t inputSize, unsigned int* histo){
    // Code from lecture slides
    __shared__ unsigned int histo_private[HISTO_WIDTH * HISTO_HEIGHT];

    if(threadIdx.x < HISTO_WIDTH * HISTO_HEIGHT){
        histo_private[threadIdx.x] = 0;
    }
    
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while(i < inputSize){
        atomicAdd(&(histo_private[d_input[i]]), 1);
        i += stride;
    }
    __syncthreads();

    if(threadIdx.x < HISTO_WIDTH * HISTO_HEIGHT){
        atomicAddWithSaturation(&(histo[threadIdx.x]), histo_private[threadIdx.x], UINT8_MAXIMUM);
    }
}

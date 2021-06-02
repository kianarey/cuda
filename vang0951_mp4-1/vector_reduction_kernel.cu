#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, int n)
{
  __shared__ float partialSum[NUM_ELEMENTS];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockDim.x * blockIdx.x;

  partialSum[t] = g_data[start + t];
  if((start + blockDim.x + t) < n){
    partialSum[blockDim.x + t] = g_data[start + blockDim.x + t];
  }
  __syncthreads();

  for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1){
    if(t < stride){
      partialSum[t] += partialSum[t + stride];
      __syncthreads();
    }
  }

  if(t == 0){
    g_data[0] = partialSum[0];
  }

}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_

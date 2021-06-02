#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 256 // section size


__global__ void scan(unsigned int *output, unsigned int *input, unsigned int *block_sums, int numElements)
{
  __shared__ unsigned int mem[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int i = 2 * blockIdx.x * blockDim.x + t;

  //Load first half into shared memory
  if(i < numElements){
    mem[t] = input[i];
  }
  else{
    mem[t] = 0;
  }

  //Load in second half
  if(i + BLOCK_SIZE < numElements){
    mem[BLOCK_SIZE + t] = input[i + BLOCK_SIZE];
  }
  else{
    mem[BLOCK_SIZE + t] = 0;
  }
  __syncthreads();

  //If t is the first thread, save the last element in shared mem
  //This will be used later
  unsigned int last;
  if(t == 0){
    last = mem[2 * BLOCK_SIZE - 1];
  }
  __syncthreads();
  
  //Reduction starts
  for(unsigned int stride = 1; stride < BLOCK_SIZE; stride *=2){
    __syncthreads();
    int index = (t + 1) * 2 * stride - 1;
    if (index < (2 * BLOCK_SIZE)){
      mem[index] += mem[index - stride];
    }
  }

  //Post reduction
  //If this is 0th thread, store last element of shared memory as 0,
  //If this is 0th block, assign 0 to the 0th element of block_sums 
  if(t == 0){
    mem[2 * BLOCK_SIZE - 1] = 0;
    if(blockIdx.x == 0){
      block_sums[blockIdx.x] = mem[2 * BLOCK_SIZE - 1];
    }
  }
  
  //for (unsigned int stride = BLOCK_SIZE/4; stride > 0; stride /= 2){
  //__syncthreads();
  //int index = (t + 1) * stride * 2 - 1;
  //if(index + stride < BLOCK_SIZE){
  //  mem[index + stride] += mem[index];
  //}
  //} 

  for(unsigned int stride = 2 * BLOCK_SIZE; stride > 0; stride /=2){
    __syncthreads();
    int index = (t + 1) * stride * 2 - 1;
    if(index < (2 * BLOCK_SIZE)){ // example in book is not correct here
      float val = mem[index];
      mem[index] += mem[index - stride];
      mem[index - stride] = val;
    }
  }
  __syncthreads();

  //Output the results from shared mem
  //First half
  if(i < numElements){
    output[i] = mem[t];
  }
  else{
    output[i] = 0;
  }
  __syncthreads();
  
  //Second half
  if(i + BLOCK_SIZE < numElements){
    output[i + BLOCK_SIZE] = mem[t + BLOCK_SIZE];
  }
  else{
    output[i + BLOCK_SIZE] = 0;
  }
  __syncthreads();

  //If this is the 0th thread, add the last element to the last element of shared mem
  //Assign this to the first element of the block  
  if(t == 0){
    block_sums[blockIdx.x] = mem[2 * BLOCK_SIZE - 1] + last;
  }
}


//Add the block sums
__global__ void sum(unsigned int *output, unsigned int *input, int numElements)
{
  int i = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(i < numElements){
    output[i] += input[blockIdx.x]; // add the element at index blockIdx.x of input to ouput indexed at i
    output[i + BLOCK_SIZE] += input[blockIdx.x]; // do the same for the second half of output
  }
}

//Lastly, the recursive function that call the above function recursively
void recursiveScan(unsigned int *output, int numElements)
{
  
  dim3 dimGrid;
  dimGrid.x = ceil(numElements/(BLOCK_SIZE * 2.0));
  dimGrid.y = 1;
  dimGrid.z = 1;
  
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  
  unsigned int *block_sums;
  unsigned int size = sizeof(unsigned int) * dimGrid.x;
  cudaMalloc((void**) &block_sums, size);

  scan<<<dimGrid, dimBlock>>>(output, output, block_sums, numElements);

  //recursive call to itself
  if(dimGrid.x > 1){
    recursiveScan(block_sums, dimGrid.x);
    sum<<<dimGrid, dimBlock>>>(output, block_sums, numElements);
  }
  
  cudaFree(block_sums);
}


// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
  //define the block dimensions and grid dimension
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(ceil(numElements/(BLOCK_SIZE * 2.0)), 1, 1);
  
  unsigned int *block_sums = NULL;
  unsigned int size = sizeof(unsigned int) * dimGrid.x;
  cudaMalloc((void**) &block_sums, size);

  //scan
  scan<<<dimGrid, dimBlock>>>(outArray, inArray, block_sums, numElements);
  //  for(unsigned int i = 0; i < size; i++){
  //printf("block_sums[%u] = %d\n", i,block_sums[i]);
  //}
  
  //recursive scan and sum
  if(dimGrid.x > 1){
    recursiveScan(block_sums, dimGrid.x);
    sum<<<dimGrid, dimBlock>>>(outArray, block_sums, numElements);
  }

  cudaFree(block_sums);

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_

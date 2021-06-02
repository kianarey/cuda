# Applied Parallel Programming using CUDA/C++ Programming
All assignments completed by me for course EE5351 Applied Parallel Programming at the University of Minnesota-Twin Cities. All programs are complete, tested, and functional.

The GPU and CUDA versions used in this course:
- 03:00.0 VGA compatible controller: NVIDIA Corporation GF106GL [Quadro2000] (rev a1)
- 04:00.0 VGA compatible controller: NVIDIA Corporation GF100 [GeForceGTX 480] (rev a3)
- 22:00.0 VGA compatible controller: NVIDIA Corporation GP104 [GeForceGTX 1080] (rev a1)

### Machine Problem 1 - Matrix Multiplication
The purpose of this machine problem is to learn the computation pattern for parallel matrix multiplication. The `MatrixMulOnDevice()` function in `matrixmul.cu` and the `MatrixMulKernel()` function in `matrixmul_kernel.cu` are edited to complete the functionality of the matrix multiplication on the device. Source code elsewhere are previously provided. The size of the matrix is defined such that one thread block will be sufficient to compute the entire solution matrix.

### Machine Problem 2 - Tiled Matrix Multiplication
The purpose of this machine problem is to learn about shared memory tiling and apply it to a matrix multiplication problem to alleviate the memory bandwidth bottleneck. The `MatrixMulOnDevice()` function in `matrixmul.cu` and the `MatrixMulKernel()` function in `matrixmul_kernel.cu` are edited to complete the functionality of the matrix multiplication on the device. Source code elsewhere are previously provided. The two matrices are of any size, but one CUDA grid is guaranteed to cover the entire output matrix. 

### Machine Problem 3 - 2D Tiled Convolution
The purpose of this machine problem is to learn how constant memory and shared memory can be used to alleviate memory bandwidth bottlenecks in the context of a convolution computation. This is a tiled implementation of a matrix convolution. It will have a 5x5 convolution kernel, but will have arbitrarily sized "images." 

### Machine Problem 4-1 - Reduction
The purpose of this machine problem is to implement a work-efficient parallel reduction algorithm on the GPU. It assumes an input array of any size.

### Machine Problem 4-2 - Parallel Prefix Sum (Scan)
The purpose of this machine problem is to implement a parallel prefix sum. The algorithm is also known as "scan." Scan is a useful building block for many parallel algorithms such as radix sort, quicksort, tree operations, and histograms. While scan is an appropriate algorithm for an associative operator, in this problem, addition is used. 

### Machine Problem 5 - Histogramming
Histograms are a commonly used analysis tool in image processing and data mining application. They show the frequency of occurrence of data elements over discrete intervals, also known as bins. The purpose of this machine problem is to explore this topic. Below are key assumptions/constraints:

- The input data consists of index values to the bins.
- The input bins are NOT uniformly distributed. This non-uniformity is a large portion of what makes this problem interesting for GPU's.
- For each bin in the histogram, once the bin count reaches 255, no further incrementing will occur. This is sometimes known as a "saturating counter."

Furthermore, the kernel runtime is measured.

### Machine Problem 6 - Sparse Matrix-Vector Multiplication
The purpose of this machine problem is to understand sparse matrix storage formats and their impacts on performance using sparse matrix-vector multiplication. The formats used in this problem are Compressed Sparse Row (CSR) and Jagged Diagonal Storage (JDS). 


## How to Run Code:
Be sure to include all files within same directory before running code. Open up a new command line shell (e.g. Terminal), navigate to the directory where files are saved, and type/enter the following:

`$ make`

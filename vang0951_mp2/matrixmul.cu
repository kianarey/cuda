/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include "matrixmul_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);
bool CompareMatrices(Matrix A, Matrix B);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
int ReadParamsFile(int* params, char* file_name, int num_params);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Matrix  M;
	Matrix  N;
	Matrix  P;
	int errorM = 0, errorN = 0;

	srand(52);

	if(argc != 5 && argc != 4)
	{
		// Allocate and initialize the matrices
		M  = AllocateMatrix(rand() % 1024, rand() % 1024, 1);
		N  = AllocateMatrix(M.width, rand() % 1024, 1);
		P  = AllocateMatrix(M.height, N.width, 0);
	}
	else
	{
		// Allocate and read in matrices from disk
		int* params = (int*)malloc(3 * sizeof(int));
		unsigned data_read = ReadParamsFile(params, argv[1], 3);
		if(data_read != 3){
			printf("Error reading parameter file\n");
			return 1;
		}

		M  = AllocateMatrix(params[0], params[1], 0);
		N  = AllocateMatrix(params[1], params[2], 0);
		P  = AllocateMatrix(params[0], params[2], 0);
		unsigned sizeM = ReadFile(&M, argv[2]);
		unsigned sizeN = ReadFile(&N, argv[3]);
		if( (sizeM != M.height * M.width) || (sizeN != N.height * N.width) )
		{
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}
	//printf("M.width is %d, M.height is %d\nN.width is %d, N.height is %d\nP.width is %d, P.height is %d\n",
	//     M.width, M.height, N.width, N.height, P.width, P.height);

	// M * N on the device
    MatrixMulOnDevice(M, N, P);
    
    printf("GPU computation complete\n");
    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width, 0);
    computeGold(reference.elements, M.elements, N.elements, M.height, M.width, N.width);
    
    printf("CPU computation complete\n");

    // print results
    //for (int i = 0; i < P.height * P.width; i++){
      
    //	printf("i: %d reference = %f,  P.elements = %f\n", i, reference.elements[i],  P.elements[i]);
      
    //}

    //printf("P.width/dimBlock.x is %d\n P.height/dimBlock.y is %d\n", ceil(P.width/dimBlock.x), ceil(P.height/dimBlock.y));
    

    // check if the device result is equivalent to the expected solution
    bool res = CompareMatrices(reference, P);
    printf("Test %s\n", res ? "PASSED" : "FAILED");

    if(argc == 5)
    {
		WriteFile(P, argv[4]);
	}
	else if(argc == 2)
	{
	    WriteFile(P, argv[1]);
	}

	// Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory

    // Get device properties
    //int deviceCount;
    //cudaGetDeviceCount(&deviceCount);
    //int device;
    //for (device = 0; device < deviceCount; ++device) {
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, device);
    //printf("Device %d  has compute capability %d.%d.\n", device,
    //	     deviceProp.major, deviceProp.minor);
    //}

    // Setup the execution configuration
    
    int gridX = P.width/TILE_WIDTH;
    if(P.width % TILE_WIDTH != 0){ // If P.width is not even multiple of TILE_WIDTH
      gridX++; // P.width is even multiple of TILE_WIDTH
    }

    int gridY = P.height/TILE_WIDTH;
    if(P.height % TILE_WIDTH != 0){
      gridY++;
    }

    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(gridX, gridY, 1); //need to make sure the parameters are multiples of TILE_WIDTH
    
    //printf("P.width/dimBlock.x is %d\n P.height/dimBlock.y is %d\n", gridX, gridY);
    
    //for (int i = 0; i < M.height * M.width; i++){
    //printf("M.element[%d] = %f\n", i, M.elements[i]);
    //}

    //for(int i = 0; i < N.height * N.width; i++){
    //printf("N.element[%d] = %f\n", i, N.elements[i]);
    //}

    // Launch the device computation threads!
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    // Sync
    cudaDeviceSynchronize();
    
    // Read P from the device
    CopyFromDeviceMatrix(P, Pd);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    // don't allocate memory on option 2
    if(init == 2)
		return M;

	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand()*3 / (float)RAND_MAX);
	}
    return M;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
					cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,
					cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int data_read = M->width * M->height;
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < data_read; i++)
        fscanf(input, "%f", &(M->elements[i]));
    return data_read;
}

// Read params of input matrices
int ReadParamsFile(int* params, char* file_name, int num_params)
{
    FILE* input = fopen(file_name, "r");
    for (unsigned i = 0; i < num_params; i++)
        fscanf(input, "%d", &(params[i]));
    return num_params;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    unsigned int size = M.width * M.height;
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < size; i++) {
        fprintf(output, "%f ", M.elements[i]);
    }
}

// returns true iff A and B have same elements in same order
bool CompareMatrices(Matrix A, Matrix B) {
    unsigned int size = A.width * A.height;

    if ( (A.width != B.width) || (A.height != B.height) )
        return false;

    for (unsigned i = 0; i < size; i++)
        if (abs(A.elements[i] - B.elements[i]) > 0.0001f)
            return false;
    return true;
}

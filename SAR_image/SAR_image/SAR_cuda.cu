#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
//************************************************************************************/
// Title : Parallel Processing of SAR Signal for Image generation on CUDA Platform
//************************************************************************************/
// Program to generate processed image from raw SAR image using parallel
// programming language CUDA

// ******** Header Files ***********//
#include <stdlib.h>
#include <math.h>
// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
// *****End of Header Files***********//

// No of rows and columns and the no of points are defined
#define ROW	2048
#define COL	2048
#define NOP	2048

// Definition for matrix multiplication
// Thread block size 
#define BLOCK_SIZE 1

// Basic Matrix dimensions 
// (chosen as multiples of the thread block size for simplicity)
#define WA (24  * BLOCK_SIZE) // Matrix A width
#define HA (7 * BLOCK_SIZE) // Matrix A height
#define WB (3584  * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

// Used in matrixMul kernel
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

//************************************************************************************/
// Start of Function prototype
// Function to perform block correlation on input data vector
void block_corr(int, int, int, cufftComplex *, cufftComplex *, cufftComplex *, cufftComplex *, cufftComplex *, cuComplex *, cuComplex *,
	cuComplex *, cuComplex *, cuComplex *, cuComplex *, cuComplex *, cuComplex *);

// Function to Flip matrix in up/down direction
void flipud(cuComplex **, int, int, cuComplex **);

// Function to Flip matrix in left/right direction
void fliplr(cuComplex *, int, cuComplex *);

// Function to swap data in blocks of 512
__global__ void swap_data(cufftComplex *, cufftComplex *, int);

// Function to launch kernel for range data processing
__global__ void process_range(cuComplex *, cuComplex *, cuComplex *, cuComplex *, int);

// Function to launch kernel for azimuth data processing
__global__ void process_az(cuComplex *, cuComplex *, cuComplex *, int);

// Function to matrix multiplication kernel 
__global__ void matrixMul(cuComplex*, cuComplex*, cuComplex*, int, int);

// Function to normalize azimuth data after ifft
__global__ void divide_by_N_azimuth(cufftComplex *, int);

// Function to normalize range data after ifft
__global__ void divide_by_N_range(cufftComplex *, int);

// Function to populate the C matrix with data
void populate_C(cuComplex *, int, cuComplex*);

// End of function prototype
//************************************************************************************/

// Start of main() function
int main()
{
	int i, j, k, w, m, flag = 0;
	int N = NOP / 4;
	double length = 1349;
	m = (int)floor(length * 8 / NOP);
	m = (int)(m + 1)*(NOP / 8);
	cudaError_t  cuda_error;

	/*Reading Raw Image file from image.txt and storing it in a[*][*] */
	int row = ROW / 2;                     // this is done to read only half the image as per logic
	int col = COL;
	FILE *fp1;

	fp1 = fopen("image.txt", "r");
	fseek(fp1, row * COL * 4 * sizeof(cuComplex), SEEK_SET);

	cuComplex **a = (cuComplex **)calloc(row, sizeof(cuComplex *));

	printf("Reading complex image\n");
	for (i = 0; i < row; i++)
	{
		a[i] = (cuComplex *)calloc(col, sizeof(cuComplex));
		for (j = 0; j < col; j++)
		{
			fscanf(fp1, "%f%f", &a[i][j].x, &a[i][j].y);
		}
	}
	fclose(fp1);
	printf("finished reading image\n");

	/*============================================================================*/
	/*                        RANGE IMAGE PROCESSING                              */
	/*============================================================================*/
	/* Reading transmit data from x_data.txt and storing it to x_data[*] */
	fp1 = fopen("x_data.txt", "r");
	fseek(fp1, 0, SEEK_END);

	int fileLen;
	fileLen = ftell(fp1);
	fileLen = fileLen / (4 * sizeof(cuComplex));	// 4*sizeof(cuComplex) is the size of one complex data
	fseek(fp1, 0, SEEK_SET);

	cuComplex * x_data = (cuComplex *)calloc(fileLen, sizeof(cuComplex));
	for (i = 0; i < fileLen; i++)
	{
		fscanf(fp1, "%f%f", &x_data[i].x, &x_data[i].y);
	}
	fclose(fp1);

	/*Calculate cuFFT of transmit data ( x_data[*] ) */

	// Rearrange the block of x_data : 3-1 to 1-3 blocks ... one block of 512 points.
	cuComplex * tx_temp = (cuComplex *)calloc(3 * N, sizeof(cuComplex));
	j = 2 * N;
	for (i = 0; i < N; i++)
	{
		tx_temp[i].x = x_data[j].x;
		tx_temp[i].y = x_data[j].y;

		tx_temp[i + N].x = x_data[j - N].x;
		tx_temp[i + N].y = x_data[j - N].y;

		tx_temp[i + (2 * N)].x = x_data[j - (2 * N)].x;
		tx_temp[i + (2 * N)].y = x_data[j - (2 * N)].y;

		j++;
	}
	// Making 12 blocks of data from given 3 blocks 
	// each block is of size 1024 = 2*512 = 2 * Previous block size

	cuComplex * tx_new = (cuComplex *)calloc(12 * 2 * N, sizeof(cuComplex));
	int p = 0;
	for (w = 0; w < 4; w++)
	{
		for (i = 0; i < 3; i++)
		{
			for (k = (i*N); k < (i + 1)*N; k++)
			{
				tx_new[p] = tx_temp[k];
				p++;
			}
			for (k = (i + 1)*N; k < (i + 2)*N; k++)
			{
				tx_new[p].x = 0.0;
				tx_new[p].y = 0.0;
				p++;
			}
		}
	}

	cudaError_t error;

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Calculating cuFFFT of transmit data (tx_new[*])

	cufftComplex * d_signal_tx;
	int mem_size = sizeof(cufftComplex)*N * 12 * 2;

	// Memory allocation & Transfer on Device 
	cuda_error = cudaMalloc((void**)&d_signal_tx, mem_size);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Malloc...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}
	cuda_error = cudaMemcpy(d_signal_tx, tx_new, mem_size, cudaMemcpyHostToDevice);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Mem Copy of d_signal_tx...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Finding cuFFT by declaring a plan 	
	cufftHandle plan;
	cufftPlan1d(&plan, 2 * 512, CUFFT_C2C, 12);
	cufftExecC2C(plan, (cufftComplex *)d_signal_tx, (cufftComplex *)d_signal_tx, CUFFT_FORWARD);
	// FFT results are stored back in to d_signal_tx on device

	/*Calculate cuFFT of received data ( y_data[*] ) */
	/*y_data[*] contains SINGLE ROW of received image data (a[*][*]) */

	cuComplex * y_data = (cuComplex *)calloc(NOP, sizeof(cuComplex));
	cuComplex * corr_output = (cuComplex *)calloc(2 * NOP, sizeof(cuComplex));
	cuComplex * rx_new = (cuComplex *)calloc(N * 12 * 2, sizeof(cuComplex));

	// Passing the variable to divide by 2 on device 
	// This can be defined directly on the device - Do it later
	cufftComplex h_tmp;
	h_tmp.x = 0.5;
	h_tmp.y = 0;

	int mem_size1 = sizeof(cufftComplex);
	cufftComplex *d_tmp;
	cuda_error = cudaMalloc((void**)&d_tmp, mem_size1);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Malloc of d_out...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}
	cuda_error = cudaMemcpy(d_tmp, &h_tmp, mem_size1, cudaMemcpyHostToDevice);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Memcpy of d_tmp...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}

	// CUDA Malloc of Receive data (y_data -> d_signal_rx) & O/P data (d_out) 	

	cufftComplex * d_signal_rx, *d_out, *d_tmp_out; // d_tmp_out is used to temporary store the data on device
	cuda_error = cudaMalloc((void**)&d_signal_rx, mem_size);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Malloc of d_signal_rx...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}
	cuda_error = cudaMalloc((void**)&d_out, mem_size);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Malloc of d_out...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}
	cuda_error = cudaMalloc((void**)&d_tmp_out, mem_size);
	if (cuda_error != cudaSuccess)
	{
		printf("error in Cuda Malloc of d_tmp_out...\n");
		printf("%s\n", cudaGetErrorString(cuda_error));
	}

	// M is the number of blocks of transmit data. 	
	int M = 4;
	int row_size = M + m / N;
	int col_size = 2 * (M*m / N);

	// Defining a constant 'A' matrix 
	cuComplex * A = (cuComplex *)calloc(row_size*col_size, sizeof(cuComplex));
	A[0].x = 1;           // All imaginary values of 'A' matrix elements are zero
	for (i = 25; i < 28; i++)
		A[i].x = 1;
	for (i = 52; i < 57; i++)
		A[i].x = 1;
	for (i = 81; i < 87; i++)
		A[i].x = 1;
	for (i = 111; i < 116; i++)
		A[i].x = 1;
	for (i = 140; i < 143; i++)
		A[i].x = 1;
	A[167].x = 1;

	cuComplex *C = (cuComplex *)calloc(24 * 7 * N, sizeof(cuComplex));
	cuComplex *fft_temp = (cuComplex *)calloc(12 * 2 * N, sizeof(cuComplex));
	// memory allocation for matrix multiplication 

	// allocate host memory for matrices A and B
	unsigned int mem_size_A = sizeof(cuComplex) * WA * HA;
	unsigned int mem_size_B = sizeof(cuComplex) * WB * HB;
	unsigned int mem_size_C = sizeof(cuComplex) * WC * HC;

	// allocate device memory
	cuComplex *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_B, mem_size_B);
	cudaMalloc((void**)&d_C, mem_size_C);

	// allocate host memory for the result
	cuComplex* h_C = (cuComplex*)malloc(mem_size_C);

	// end of matrix mult memory allocation

	cuComplex **range_image = (cuComplex **)calloc(row, sizeof(cuComplex *));
	cuComplex **range_image_flip = (cuComplex **)calloc(row, sizeof(cuComplex *));

	/*Starting of Range Image Processing */

	i = 0;
	for (j = 0; j < row; j++)
	{
		for (k = 0; k < NOP; k++)
		{
			y_data[k] = cuConjf(a[j][k]);
		}

		// Block formation of receive data of block size = 1024 points
		p = 0;
		for (w = 0; w < 4; w++)
		{
			for (int x = 0; x < 3; x++)
			{
				for (k = (w*N); k < (w + 1)*N; k++)
				{
					rx_new[p] = y_data[k];
					p++;
				}
				for (k = (w + 1)*N; k < (w + 2)*N; k++)
				{
					rx_new[p].x = 0.0;
					rx_new[p].y = 0.0;
					p++;
				}
			}
		}

		// Compute Block Correlation of the Transmit & Receive data.
		cuda_error = cudaMemcpy(d_signal_rx, rx_new, mem_size, cudaMemcpyHostToDevice);
		if (cuda_error != cudaSuccess)
		{
			printf("error in Cuda Mem Copy of d_signal_rx...\n");
			printf("%s\n", cudaGetErrorString(cuda_error));
		}

		block_corr(flag, N, m, d_signal_tx, d_signal_rx, d_out, d_tmp, d_tmp_out, corr_output, A, C, fft_temp, d_A, d_B, d_C, h_C);
		range_image[j] = (cuComplex *)calloc(col, sizeof(cuComplex));
		range_image_flip[j] = (cuComplex *)calloc(col, sizeof(cuComplex));
		int z = NOP;
		for (k = 0; k < NOP; k++)
		{
			range_image[i][k] = corr_output[z];
			z++;
		}
		i++;
	}
	printf("Finished range image processing\n");

	/*END OF RANGE IMAGE PROCESSING*/

	/*============================================================================*/
	/*                        RANGE IMAGE PROCESSING                              */
	/*============================================================================*/

	//azimuth processing commences
	/*	fp1=fopen("range_image1.txt","r");

	for(i=0;i<row;i++)
	{
	for(j=0;j<col;j++)
	{
	fscanf(fp1,"%f%f",&range_image[i][j].x,&range_image[i][j].y);
	}
	}
	fclose(fp1);*/
	flipud(range_image, row, col, range_image_flip);
	flag = 1;
	if (flag == 1)
	{
		int nrow = 1024;
		double L = 701;
		N = nrow / 4;
		m = (int)floor(L / N);
		m = (int)(m + 1)*N;
		fp1 = fopen("x_data_az.txt", "r");
		fseek(fp1, 0, SEEK_END);
		fileLen = ftell(fp1);
		fileLen = fileLen / (4 * sizeof(cuComplex));	// 4*sizeof(cuComplex) is the size of one complex data
		fseek(fp1, 0, SEEK_SET);
		cuComplex * x_data = (cuComplex *)calloc(fileLen, sizeof(cuComplex));

		for (i = 0; i < fileLen; i++)
		{
			fscanf(fp1, "%f%f", &x_data[i].x, &x_data[i].y);
		}
		fclose(fp1);
		cuComplex * x_flip_data = (cuComplex *)calloc(N, sizeof(cuComplex));
		cuComplex * x_temp_data = (cuComplex *)calloc(N, sizeof(cuComplex));
		cuComplex * rx_new = (cuComplex *)calloc(24 * N, sizeof(cuComplex));
		cuComplex * tx_temp = (cuComplex *)calloc(3 * N, sizeof(cuComplex));
		cuComplex * tx_new = (cuComplex *)calloc(12 * 2 * N, sizeof(cuComplex));
		j = 2 * N;
		for (i = 0; i < N; i++)
		{
			tx_temp[i].x = x_data[j].x;
			tx_temp[i].y = x_data[j].y;

			tx_temp[i + N].x = x_data[j - N].x;
			tx_temp[i + N].y = x_data[j - N].y;

			tx_temp[i + (2 * N)].x = x_data[j - (2 * N)].x;
			tx_temp[i + (2 * N)].y = x_data[j - (2 * N)].y;

			j++;
		}

		mem_size = sizeof(cufftComplex)*N * 12 * 2;
		int p = 0;
		int z, q;
		for (w = 0; w < 4; w++)
		{
			for (i = 0; i < 3; i++)
			{
				q = 0;
				for (z = i*N; z < (i + 1)*N; z++)
				{
					x_temp_data[q] = cuConjf(tx_temp[z]);
					q++;
				}
				fliplr(x_temp_data, N, x_flip_data);
				for (k = 0; k < N; k++)
				{
					tx_new[p] = x_flip_data[k];
					p++;
				}
				for (k = 0; k < N; k++)
				{
					tx_new[p].x = 0.0;
					tx_new[p].y = 0.0;
					p++;
				}
			}
		}
		cuda_error = cudaMemcpy(d_signal_tx, tx_new, mem_size, cudaMemcpyHostToDevice);
		if (cuda_error != cudaSuccess)
		{
			printf("error in Cuda Mem Copy...\n");
			printf("%s\n", cudaGetErrorString(cuda_error));
		}
		cufftPlan1d(&plan, 512, CUFFT_C2C, 12);
		cufftExecC2C(plan, (cufftComplex *)d_signal_tx, (cufftComplex *)d_signal_tx, CUFFT_INVERSE);

		// allocate host memory for matrices A and B

		unsigned int size_A = WA * HA;
		unsigned int mem_size_A = sizeof(cuComplex) * size_A;

		unsigned int size_B = (WB / 2) * HB;
		unsigned int mem_size_B = sizeof(cuComplex) * size_B;

		// allocate device memory
		cuComplex* d_A;
		cudaMalloc((void**)&d_A, mem_size_A);
		cuComplex* d_B;
		cudaMalloc((void**)&d_B, mem_size_B);

		// allocate device memory for result
		unsigned int size_C = (WC / 2) * HC;
		unsigned int mem_size_C = sizeof(cuComplex) * size_C;
		cuComplex* d_C;
		cudaMalloc((void**)&d_C, mem_size_C);
		// allocate host memory for the result
		cuComplex* h_C = (cuComplex*)malloc(mem_size_C);
		// end of matrix multi memory allocation	

		i = 0;
		for (j = 0; j < col; j++)
		{
			for (k = 0; k < row; k++)
			{
				y_data[k] = range_image_flip[k][j];
			}
			p = 0;
			for (w = 0; w < 4; w++)
			{
				for (int x = 0; x < 3; x++)
				{
					for (k = (w*N); k < (w + 1)*N; k++)
					{
						rx_new[p] = y_data[k];
						p++;
					}
					for (k = (w + 1)*N; k < (w + 2)*N; k++)
					{
						rx_new[p].x = 0.0;
						rx_new[p].y = 0.0;
						p++;
					}
				}
			}
			cuda_error = cudaMemcpy(d_signal_rx, rx_new, mem_size, cudaMemcpyHostToDevice);
			if (cuda_error != cudaSuccess)
			{
				printf("error in Cuda Mem Copy...\n");
				printf("%s\n", cudaGetErrorString(cuda_error));
			}

			block_corr(flag, N, m, d_signal_tx, d_signal_rx, d_out, d_tmp, d_tmp_out, corr_output, A, C, fft_temp, d_A, d_B, d_C, h_C);

			w = N;

			for (k = 0; k < 4 * N; k++)
			{
				range_image[k][i] = corr_output[w];
				w++;
			}
			i++;
		}
		// Record the stop event
		error = cudaEventRecord(stop, NULL);

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		// Wait for the stop event to complete
		error = cudaEventSynchronize(stop);

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}

		float msecTotal = 0.0f;
		error = cudaEventElapsedTime(&msecTotal, start, stop);

		if (error != cudaSuccess)
		{
			fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
			exit(EXIT_FAILURE);
		}
		printf("\nProcessing time: %f (ms)\n\n", msecTotal);

		// data is written in azimuth file seperately bcos azimuth image is stored in col major format 
		// and not in row major format
		fp1 = fopen("azimuth_image.txt", "w");
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < col; j++)
			{
				fprintf(fp1, "%lg\t", cuCabsf(range_image[i][j]));
			}
		}
		fclose(fp1);
	}

	// Memory free allocated
	for (i = 0; i < row; i++)
	{
		free(range_image[i]);
		free(range_image_flip[i]);
	}
	free(range_image);
	free(range_image_flip);
	free(C);
	free(A);
	free(fft_temp);
	free(h_C);
	cudaFree(d_signal_tx);
	cudaFree(d_signal_rx);
	cudaFree(d_out);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
// End of main()

// Start of user defined functions 
// fliplr function implementation of one row and 'n' col of data
void fliplr(cuComplex *in, int col, cuComplex *out)
{
	int i, k;
	if (col % 2 != 0)
	{
		k = col;
		for (i = 0; i < col / 2; i++)
		{
			out[i] = in[k - 1 - i];
		}
		k = col - 1;
		for (i = col / 2; i < col; i++)
		{
			out[i] = in[k - i];
		}
	}
	else
	{
		k = col;
		for (i = 0; i < col; i++)
		{
			out[i] = in[k - 1 - i];
		}
	}
}

void flipud(cuComplex **in, int row, int col, cuComplex **out)
{
	int i, j, k;
	if (row % 2 != 0)
	{
		k = row;
		for (i = 0; i < row / 2; i++)
		{
			for (j = 0; j < col; j++)
			{
				out[i][j] = in[k - 1 - i][j];
			}
		}
		k = row - 1;
		for (i = row / 2 + 1; i < row; i++)
		{
			for (j = 0; j < col; j++)
			{
				out[i][j] = in[k - i][j];
			}
		}
		for (j = 0; j < col; j++)
		{
			out[row / 2][j] = in[row / 2][j];
		}
	}
	else
	{
		k = row;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < col; j++)
			{
				out[i][j] = in[k - 1 - i][j];
			}
		}
	}

}

__global__ void matrixMul(cuComplex* C, cuComplex* A, cuComplex* B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	cuComplex Csub;
	Csub.x = 0;
	Csub.y = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ cuComplex As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ cuComplex Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		AS(ty, tx) = A[a + wA * ty + tx];
		BS(ty, tx) = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub = cuCaddf(Csub, cuCmulf(AS(ty, k), BS(k, tx)));
		//Csub += AS(ty, k) * BS(k, tx);

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

__global__ void divide_by_N_azimuth(cufftComplex *d_out, int N)
{
	//step 1: d_out signal normalization, after cuFFT inverse of d_out from host.

	int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (thread_ID < N)
	{
		d_out[thread_ID].x = d_out[thread_ID].x / (2 * 256);
		d_out[thread_ID].y = d_out[thread_ID].y / (2 * 256);
	}
	__syncthreads();
}
__global__ void divide_by_N_range(cufftComplex *d_out, int N)
{
	//step 1: d_out signal normalization, after cuFFT inverse of d_out from host.

	int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (thread_ID < N)
	{
		d_out[thread_ID].x = d_out[thread_ID].x / (2 * 512);
		d_out[thread_ID].y = d_out[thread_ID].y / (2 * 512);
	}
	__syncthreads();
}
__global__ void process_range(cuComplex *tx_new, cuComplex *rx_new, cuComplex *d_out, cuComplex *d_tmp, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	if (index < N)
	{
		d_out[index] = cuCmulf(d_tmp[0], cuCmulf(cuConjf(tx_new[index]), rx_new[index]));
	}
	__syncthreads();
}

__global__ void swap_data(cufftComplex *d_tmp_out, cufftComplex *d_out, int N)
{
	int thread_ID = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	if (thread_ID < N)
	{
		if (blockIdx.x % 2 == 0)
		{
			d_tmp_out[thread_ID] = d_out[thread_ID + 512];
		}
		else
		{
			d_tmp_out[thread_ID] = d_out[thread_ID - 512];
		}
	}
	__syncthreads();
}
__global__ void process_az(cuComplex *tx_new, cuComplex *rx_new, cuComplex *d_out, int N)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	if (index < N)
	{
		d_out[index] = cuConjf(cuCmulf(cuConjf(rx_new[index]), tx_new[index]));
	}

	__syncthreads();
}
void populate_C(cuComplex * C, int N, cuComplex* fft_temp)
{
	int i, j = 0;
	int w;

	j = 0;
	w = 0;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 8 * N;
	w = N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 15 * N;
	w = 2 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 22 * N;
	w = 6 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 30 * N;
	w = 3 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 37 * N;
	w = 7 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 44 * N;
	w = 4 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 51 * N;
	w = 8 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 58 * N;
	w = 12 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 66 * N;
	w = 5 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 73 * N;
	w = 9 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 80 * N;
	w = 10 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 87 * N;
	w = 13 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 94 * N;
	w = 14 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 101 * N;
	w = 18 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 109 * N;
	w = 11 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 116 * N;
	w = 15 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 123 * N;
	w = 16 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 130 * N;
	w = 19 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 137 * N;
	w = 20 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 145 * N;
	w = 17 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 152 * N;
	w = 21 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 159 * N;
	w = 22 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}
	j = 167 * N;
	w = 23 * N;
	for (i = 0; i < N; i++)
	{
		C[j] = fft_temp[i + w];
		j++;
	}

}

void block_corr(int flag, int N, int m, cufftComplex *d_signal_tx, cufftComplex *d_signal_rx, cufftComplex *d_out,
	cufftComplex *d_tmp, cufftComplex *d_tmp_out, cuComplex *corr_output, cuComplex *A, cuComplex *C, cuComplex *fft_temp,
	cuComplex *d_A, cuComplex *d_B, cuComplex *d_C, cuComplex *h_C)
{

	cudaError_t  cuda_error;

	int mem_size = sizeof(cufftComplex)*N * 12 * 2;
	cufftHandle plan1;

	dim3  dim_block(512, 1, 1);
	dim3  dim_grid(12, 2, 1);

	if (flag == 0)
	{
		cufftPlan1d(&plan1, 2 * 512, CUFFT_C2C, 12);
		cufftExecC2C(plan1, (cufftComplex *)d_signal_rx, (cufftComplex *)d_signal_rx, CUFFT_FORWARD);

		process_range << <dim_grid, dim_block >> >(d_signal_tx, d_signal_rx, d_out, d_tmp, 24 * N);

		cudaThreadSynchronize();
		cuda_error = cudaGetLastError();
		if (cuda_error != cudaSuccess)
		{
			printf("error in launching kernel processdata_kernel.\n");
			printf("%s\n", cudaGetErrorString(cuda_error));
		}

		cufftPlan1d(&plan1, 2 * 512, CUFFT_C2C, 12);
		cufftExecC2C(plan1, (cufftComplex *)d_out, (cufftComplex *)d_out, CUFFT_INVERSE);

		divide_by_N_range << <dim_grid, dim_block >> >(d_out, 24 * N);
		cudaThreadSynchronize();
		cuda_error = cudaGetLastError();
		if (cuda_error != cudaSuccess)
		{
			printf("error in launching kernel process_range_kernel.\n");
			printf("%s\n", cudaGetErrorString(cuda_error));
		}

		// kernel call to swap data after ifft
		swap_data << <dim_grid, dim_block >> >(d_tmp_out, d_out, 24 * N);
		cudaThreadSynchronize();
		cufftDestroy(plan1);

		cudaMemcpy(fft_temp, d_tmp_out, mem_size, cudaMemcpyDeviceToHost);

	}
	if (flag == 1)
	{
		cufftPlan1d(&plan1, 512, CUFFT_C2C, 12);
		cufftExecC2C(plan1, (cufftComplex *)d_signal_rx, (cufftComplex *)d_signal_rx, CUFFT_FORWARD);

		process_az << <12, 512 >> >(d_signal_tx, d_signal_rx, d_out, 24 * N);
		cudaThreadSynchronize();

		cufftExecC2C(plan1, (cufftComplex *)d_out, (cufftComplex *)d_out, CUFFT_INVERSE);

		divide_by_N_azimuth << <dim_grid, dim_block >> >(d_out, 24 * N);
		cudaThreadSynchronize();
		// we need not swipe back the data which was not done in range processing
		// bcos the swipe data ops is not required in azimuth processing
		cudaMemcpy(fft_temp, d_out, mem_size, cudaMemcpyDeviceToHost);

	}
	int i, j = 0, k;

	int M = 4;
	int row_size = M + m / N;
	int col_size = 2 * (M*m / N);
	populate_C(C, N, fft_temp);

	// allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(cuComplex) * size_A;

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(cuComplex) * size_B;

	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(cuComplex) * size_C;
	if (flag == 1)
	{
		mem_size_B = mem_size_B / 2;
		mem_size_C = mem_size_C / 2;
	}
	cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, C, mem_size_B, cudaMemcpyHostToDevice);

	// setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(WC / threads.x, HC / threads.y);

	if (flag == 1)
	{
		matrixMul << < grid, threads >> >(d_C, d_A, d_B, WA, WB / 2);
		cudaThreadSynchronize();
	}
	else
	{
		matrixMul << < grid, threads >> >(d_C, d_A, d_B, WA, WB);
		cudaThreadSynchronize();
	}

	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	k = 0;
	for (i = 0; i < 7 * N; i += N)
	{
		for (j = i + N; j < i + 2 * N; j++)
		{
			corr_output[j] = h_C[k];
			k++;
		}
		k = k + 7 * N;
	}
}
// End of user defined functions 

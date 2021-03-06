/*
  GPU Memory Transfer Benchmark Program - benchmark.cu
  Written by Cody Rivera

  Test controller and CUDA kernel definitions

  Uses CUDA events to time ops.
 */

#include <cstdio>
#include <cmath>
#include "cuda_runtime.h"

#include "benchmark.cuh"

bool selectDevices(int&, int&);

void runBenchmarks(BenchmarkRates& rates, int blockSize, int numBlocks)
{
    // Finds two linked devices
    int firstDeviceID, secondDeviceID;
    bool multipleDevices = selectDevices(firstDeviceID, secondDeviceID);
    // If multiple GPUs cannot be used, 
    if (!multipleDevices)
    {
        fprintf(stderr, "No GPUs can be paired, and the benchmark cannot be run\n");
        exit(1);
    }

    int blockInts = (int) ceil(blockSize / 4.0);
    // Allocates and sets host memory
    int* hostMem = (int*) malloc(blockInts * sizeof(int));
    if (hostMem == NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        exit(1);
    } 
    for (int i = 0; i < blockInts; i++)
    {
        // Populates with something
        hostMem[i] = i % 1024;
    }

    CUDA_ASSERT(cudaSetDevice(firstDeviceID));
    // Allocates device memory
    int* devGlobalMem;
    cudaMalloc((void**) &devGlobalMem, blockInts * sizeof(int));
    if (devGlobalMem == NULL)
    {
        fprintf(stderr, "Not enough GPU memory\n");
        exit(1);
    }

    // Allocates second batch of device memory
    int* devGlobalMem2;
    cudaMalloc((void**) &devGlobalMem2, blockInts * sizeof(int));
    if (devGlobalMem2 == NULL)
    {
        fprintf(stderr, "Not enough GPU memory\n");
        exit(1);
    }

    CUDA_ASSERT(cudaSetDevice(secondDeviceID));
    // Allocates device memory on second GPU
    int* devGlobalMemOtherGPU;
    cudaMalloc((void**) &devGlobalMemOtherGPU, blockInts * sizeof(int));
    if (devGlobalMemOtherGPU == NULL)
    {
        fprintf(stderr, "Not enough GPU memory\n");
        exit(1);
    }
    CUDA_ASSERT(cudaSetDevice(firstDeviceID));
    
    // Set up Cuda Event Timer
    cudaEvent_t start, stop;
    float time;

    CUDA_ASSERT(cudaEventCreate(&start));
    CUDA_ASSERT(cudaEventCreate(&stop));
    
    // CUDA call variables
    int cudaBlocks = (int) ceil(blockInts / 512.0);
    int cudaThreads = 512;

    // TEST 1 - CPU to GPU
    CUDA_ASSERT(cudaEventRecord(start));
    for (int i = 0; i < numBlocks; i++)
    {
        cudaMemcpy((void*) devGlobalMem, (void*) hostMem, 
                   blockInts * sizeof(int), cudaMemcpyHostToDevice);
    }
    CUDA_ASSERT(cudaEventRecord(stop));
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.CPUtoGPU = ((long) blockSize * numBlocks) / (0.001 * time);

    // TEST 2 - GPU to CPU
    CUDA_ASSERT(cudaEventRecord(start));
    for (int i = 0; i < numBlocks; i++)
    {
        cudaMemcpy((void*) hostMem, (void*) devGlobalMem, 
                   blockInts * sizeof(int), cudaMemcpyDeviceToHost);
    }
    CUDA_ASSERT(cudaEventRecord(stop));
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.GPUtoCPU = ((long) blockSize * numBlocks) / (0.001 * time);
    
    /*
    // Additional Test - GPU to GPU Async
    CUDA_ASSERT(cudaDeviceEnablePeerAccess(secondDeviceID, 0));
    CUDA_ASSERT(cudaEventRecord(start));
    for (int i = 0; i < numBlocks; i++)
    {
        cudaMemcpyPeerAsync(devGlobalMemOtherGPU, secondDeviceID,
                       devGlobalMem, firstDeviceID, blockInts * sizeof(int));
    }
    CUDA_ASSERT(cudaEventRecord(stop));
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.GPUtoGPUAsync = ((long) blockSize * numBlocks) / (0.001 * time);
    */

    // Additional Test - GPU to GPU
    CUDA_ASSERT(cudaEventRecord(start));
    for (int i = 0; i < numBlocks; i++)
    {
        cudaMemcpy((void*) devGlobalMemOtherGPU, 
                   (void*) devGlobalMem, blockInts * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    CUDA_ASSERT(cudaEventRecord(stop));
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.GPUtoGPU = ((long) blockSize * numBlocks) / (0.001 * time);

    // Additional Test - Global to Global
    CUDA_ASSERT(cudaEventRecord(start));
    for (int i = 0; i < numBlocks; i++)
    {
        cudaMemcpy((void*) devGlobalMem2, (void*) devGlobalMem, 
                   blockInts * sizeof(int), cudaMemcpyDeviceToDevice);
    }
    CUDA_ASSERT(cudaEventRecord(stop));
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    // Account for bidirectional
    rates.globalToGlobal = ((long) 2 * blockSize * numBlocks) / (0.001 * time);
        
    // TEST 3 - Global to Shared
    CUDA_ASSERT(cudaEventRecord(start));
    globalToSharedTest<<<cudaBlocks, cudaThreads, cudaThreads * sizeof(int)>>>(devGlobalMem, blockInts, numBlocks); 
    CUDA_ASSERT(cudaEventRecord(stop));
    CUDA_ASSERT(cudaGetLastError());
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.globalToShared = ((long) blockSize * numBlocks) / (0.001 * time);
   
    // TEST 4 - Shared to Global
    CUDA_ASSERT(cudaEventRecord(start));
    sharedToGlobalTest<<<cudaBlocks, cudaThreads, cudaThreads * sizeof(int)>>>(devGlobalMem, blockInts, numBlocks); 
    CUDA_ASSERT(cudaEventRecord(stop));
    CUDA_ASSERT(cudaGetLastError());
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.sharedToGlobal = ((long) blockSize * numBlocks) / (0.001 * time);
   
    // TEST 5 - Shared to Shared
    CUDA_ASSERT(cudaEventRecord(start));
    // Double the shared memory needs to be allocated
    sharedToSharedTest<<<cudaBlocks, cudaThreads, (2 * cudaThreads + 1) * sizeof(int)>>>(numBlocks); 
    CUDA_ASSERT(cudaEventRecord(stop));
    CUDA_ASSERT(cudaGetLastError());
    
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop));
    rates.sharedToShared = ((long) blockSize * numBlocks) / (0.001 * time);

    // Destroys events
    CUDA_ASSERT(cudaEventDestroy(start));
    CUDA_ASSERT(cudaEventDestroy(stop));
    
    // Deallocates memory
    free((void*) hostMem);
    cudaFree((void*) devGlobalMem);
    cudaFree((void*) devGlobalMem2);
    cudaFree((void*) devGlobalMemOtherGPU);
}


/*
  Helper function to select a pair of GPUs for the benchmark
 */
bool selectDevices(int& firstID, int& secondID)
{
    int numDevices, flag;
    CUDA_ASSERT(cudaGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numDevices; j++)
        {
            CUDA_ASSERT(cudaDeviceCanAccessPeer(&flag, i, j));
            if (flag == 1)
            {
                firstID = i;
                secondID = j;
                return true;
            }
        }
    }
    return false;
}


/* 
   GPU kernel for async GPU-GPU transfers
*/
__global__ void asyncGPUtoGPU(int* dest, int* source, int count)
{
    const int thread = threadIdx.x + (blockDim.x * blockIdx.x);
    if (thread < count)
    {
        dest[thread] = source[thread];
    }
}

/*
  GPU kernel for Global to Shared Copying -- One block is copied in parallel
  depending on block size
 */
__global__ void globalToSharedTest(int* devGlobalMem, int blockInts, int numBlocks)
{
    // Size = blockDim.x
    extern __shared__ int sharedMem[];

    const int thread = threadIdx.x + (blockDim.x * blockIdx.x);

    // Populates shared memory and syncs before writes, just to mirror others
    sharedMem[threadIdx.x] = thread;
    __syncthreads();

    for (int i = 0; i < numBlocks; i++)
    {
        if (thread < blockInts)
        {
            sharedMem[threadIdx.x] = devGlobalMem[thread];
        }
    }
    __syncthreads();
}


/*
  GPU kernel for Shared to Global Copying -- One block is copied in parallel
  depending on block size
 */
__global__ void sharedToGlobalTest(int* devGlobalMem, int blockInts, int numBlocks)
{
    // Size = blockDim.x
    extern __shared__ int sharedMem[];

    const int thread = threadIdx.x + (blockDim.x * blockIdx.x);
    
    // Populates shared memory and syncs before reads
    sharedMem[threadIdx.x] = thread;
    __syncthreads();
    
    for (int i = 0; i < numBlocks; i++)
    {
        if (thread < blockInts)
        {
            devGlobalMem[thread] = sharedMem[threadIdx.x];
        }
    }
}



/*
  GPU kernel for Shared to Shared Copying -- One segment is copied to another segment
 */
__global__ void sharedToSharedTest(int numBlocks)
{
    // Size = blockDim.x
    extern __shared__ int sharedMem[];

    const int thread = threadIdx.x + (blockDim.x * blockIdx.x);
    
    // Populates shared memory and syncs before writes
    sharedMem[threadIdx.x] = thread;
    __syncthreads();
    
    for (int i = 0; i < numBlocks; i++)
    {
        sharedMem[threadIdx.x + blockDim.x + 1] = sharedMem[threadIdx.x];
    }
    __syncthreads();
}



/*
  GPU Memory Transfer Benchmark Program - benchmark.cuh
  Written by Cody Rivera
  
  Header file for CUDA kernels and supporting structures.
 */

#ifndef _benchmark_CUH
#define _benchmark_CUH

#include <cstdio>
#include "cuda_runtime.h"

/*
  Structure to store raw speed results
 */
struct BenchmarkRates
{
    double CPUtoGPU;
    double GPUtoCPU;
    double globalToShared;
    double sharedToGlobal;
    double sharedToShared;
};


// Benchmark run function
extern void runBenchmarks(BenchmarkRates&, int, int);

// CUDA kernels
__global__ void globalToSharedTest(int*, int, int);
__global__ void sharedToGlobalTest(int*, int, int);
__global__ void sharedToSharedTest(int);


// CUDA error handling -- Based on online solutions
#define CUDA_ASSERT(C) \
    {                                               \
        cudaError_t e = C;                          \
        if (e != cudaSuccess)                       \
        {                                           \
            fprintf(stderr, "CUDA Error: %s on %d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                    \
        }                                           \
    }

#endif
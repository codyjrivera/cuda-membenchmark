/*
  GPU Memory Transfer Benchmark Program - benchmark.cpp
  Written by Cody Rivera

  Processes command-line arguments, calls tests, and displays
  results.
 */

#include <cstdio>
#include <cmath>
#include <cctype>
#include "benchmark.cuh"



int sizeToBlocks(int, double, char);
void printNormalizedSize(double);

/*
  main - Processes command line arguments and prepares tests
 */
int main(int argc, char** argv)
{
    // To hold command-line arguments
    BenchmarkRates rawRates;
    double blockSizeF, dataSize;
    int blockSize, numBlocks;
    char bunit, unit;

    // Processes command-line arguments
    switch (argc)
    {
    case 3:
        // ./a.out blockSize numBlocks
        if (!sscanf(argv[1], "%d", &blockSize) || blockSize <= 0)
        {
            fprintf(stderr, "Invalid Block Size: %s\n", argv[1]);
            return 1;
        }
        if (!sscanf(argv[2], "%d", &numBlocks) || numBlocks <= 0)
        {
            fprintf(stderr, "Invalid Block Count: %s\n", argv[2]);
            return 1;
        }
        break;
    case 4:
        // ./a.out blockSize dataSize unit
        if (!sscanf(argv[1], "%d", &blockSize) || blockSize <= 0)
        {
            fprintf(stderr, "Invalid Block Size: %s\n", argv[1]);
            return 1;
        }
        if (!sscanf(argv[2], "%lf", &dataSize) || dataSize <= 0)
        {
            fprintf(stderr, "Invalid Data Size: %s %s\n", argv[2], argv[3]);
            return 1;
        }
        // Grabs first character
        unit = toupper(argv[3][0]);
        numBlocks = sizeToBlocks(blockSize, dataSize, unit);
        if (numBlocks <= 0)
        {
            fprintf(stderr, "Invalid Data Size: %s %s\n", argv[2], argv[3]);
            return 1;
        }
        break;
    case 5:
        // ./a.out blockSizeF unit dataSize unit                                                           
        if (!sscanf(argv[1], "%lf", &blockSizeF) || blockSizeF <= 0)
        {
            fprintf(stderr, "Invalid Block Size: %s %s\n", argv[1], argv[2]);
            return 1;
        }
        // Grabs first character
        bunit = toupper(argv[2][0]);
        blockSize = sizeToBlocks(1, blockSizeF, bunit);
        if (blockSize <= 0)
        {
            fprintf(stderr, "Invalid Block Size %s %s\n", argv[1], argv[2]);
            return 1;
        }
        if (!sscanf(argv[3], "%lf", &dataSize) || dataSize <= 0)
        {
            fprintf(stderr, "Invalid Data Size: %s %s\n", argv[3], argv[4]);
            return 1;
        }
        // Grabs first character                                 
        unit = toupper(argv[4][0]);
        numBlocks = sizeToBlocks(blockSize, dataSize, unit);
        if (numBlocks <= 0)
        {
            fprintf(stderr, "Invalid Data Size: %s %s\n", argv[3], argv[4]);
            return 1;
        }
        break;
    default:
        // Help message printed when invoked incorrectly
        fprintf(stderr, "\nUsage: %s blockSize numBlocks\n", argv[0]);
        fprintf(stderr, "   or: %s blockSize dataSize unit\n", argv[0]);
        fprintf(stderr, "   or: %s blockSizeF unit dataSize unit\n\n", argv[0]);
        fprintf(stderr, "blockSize must be an integer in [1..2^31 - 1]\n");
        fprintf(stderr, "numBlocks must be an integer in [1..2^31 - 1]\n\n");
        fprintf(stderr, "blockSizeF and dataSize may be any float value greater than zero\n");
        fprintf(stderr, "unit may be any of B, K, M, G (case insensitive)\n\n");
        return 1;
        break;
    }

    // Notifies user of test start
    printf("Running benchmarks with %d blocks of %d bytes\n", numBlocks, blockSize);
    printf("Benchmark size: ");
    printNormalizedSize((double) blockSize * (double) numBlocks);
    printf("\n\n");

    // Runs Benchmarks
    runBenchmarks(rawRates, blockSize, numBlocks);

    // Prints out the results
    printf("BENCHMARK RESULTS:\n");
    printf("%*s", -60, "Write from CPU to GPU: ");
    printNormalizedSize(rawRates.CPUtoGPU);
    printf("/s\n");
    printf("%*s", -60, "Read from GPU to CPU: ");
    printNormalizedSize(rawRates.GPUtoCPU);
    printf("/s\n");
    printf("%*s", -60, "Write from one GPU to another GPU (NVLink): ");
    printNormalizedSize(rawRates.GPUtoGPU);
    printf("/s\n");
    printf("%*s", -60, "Write from Global Memory to Global Memory: ");
    printNormalizedSize(rawRates.globalToGlobal);
    printf("/s\n");
    printf("%*s", -60, "Write from Global Memory to Shared Memory: ");
    printNormalizedSize(rawRates.globalToShared);
    printf("/s\n");
    printf("%*s", -60, "Read from Shared Memory to Global Memory: ");
    printNormalizedSize(rawRates.sharedToGlobal);
    printf("/s\n");
    printf("%*s", -60, "Optimal Write within Shared Memory (no bank conflicts): ");
    printNormalizedSize(rawRates.sharedToShared);
    printf("/s\n\n");
    return 0;
}



/*
  sizeToBlocks - Converts a certain size input to number of blocks
 */
int sizeToBlocks(int blockSize, double dataSize, char unit)
{
    double rawSize;
    // Denormalizes the data size
    switch (unit)
    {
    case 'K':
        rawSize = dataSize * 1024;
        break;
    case 'M':
        rawSize = dataSize * 1024 * 1024;
        break;
    case 'G':
        rawSize = dataSize * 1024 * 1024 * 1024;
        break;
    default:
        rawSize = dataSize;
        break;
    }
    return (int) ceil(rawSize / blockSize);
}


/*
  printNormalizedSize -- Given a double that represents an amount of data,
  prints a normalized version of this
 */
void printNormalizedSize(double size)
{
    char sizeList[] = {' ', 'K', 'M', 'G', 'T', 'P', 'E'}; 
    double mantissa = size;
    int exponent = 0;
    while (fabs(mantissa) >= 10000)
    {
        mantissa /= 1024;
        exponent++;
    }
    printf("%.4lf ", mantissa);
    if (exponent < (int)(sizeof(sizeList) / sizeof(sizeList[0])))
    {
        printf("%cB", sizeList[exponent]);
    }
    else
    {
        printf("* 2^%d B", 3 * exponent);
    }
}

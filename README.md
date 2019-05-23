# GPU memory benchmark


## Introduction

This benchmark is intended to demonstrate the difference in bandwidth between CPU-GPU
memory transfers and GPU-GPU internal memory transfers. CPU-GPU transfers are much slower
than GPU-GPU transfers. Running this benchmark on a test machine with a Nvidia v100 shows that
CPU-GPU transfers are slower than GPU-GPU transfers by a factor of 10^3. As a result, to avoid
bottlenecks and excessive transfers, GPU programs must use input data wisely.


## The Benchmark

Type 'make' to build the program.

The program can be invoked with './benchmark blockSize unit dataSize unit',
where blockSize and dataSize are positive floating point numbers, and unit
is either B - (bytes), K - (kilobytes), M - (megabytes), G - (gigabytes).

Other ways of invoking this program can be shown by typing just './benchmark'.

### Execution example

'./benchmark 128 M 2.5 G' runs the benchmark with 2.5 gigabytes of data, operated on
128 megabytes at a time.

### Sample Output

Running benchmarks with 20 blocks of 134217728 bytes
Benchmark size: 2560.0000 MB

BENCHMARK RESULTS:
Write from CPU to GPU:                                      9121.8556 MB/s
Read from GPU to CPU:                                       9464.6241 MB/s
Write from Global Memory to Shared Memory:                  15.4254 TB/s
Read from Shared Memory to Global Memory:                   11.1297 TB/s
Optimal Write within Shared Memory (no bank conflicts):     11.2131 TB/s




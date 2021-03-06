# CUDA/CPP Makefile Template

# Product Names
CUDA_OBJ = cuda.o

# Input Names
CUDA_FILES = benchmark.cu
CPP_FILES = benchmark.cpp

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -O3 -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS = 
NVCC_GENCODES = \
-gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_70,code=compute_70

# CUDA Object Files
CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++11 -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lm

# ------------------------------------------------------------------------------
# Make Rules
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ = $(notdir $(addsuffix .o, $(CPP_FILES)))

# Top level rules
all: benchmark

benchmark.cu:;

benchmark: $(OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o benchmark $(INCLUDE) $^ $(LIBS) 


# Compile C++ Source Files
%.cpp.o: %.cpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 


# Compile CUDA Source Files
%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f *.o *.test.* benchmark

.PHONY: clean

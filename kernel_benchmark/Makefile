# Copyright 2025 The ZipServ Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     := -fopenmp -O3 -march=native
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/ 
LIBRARIES := -lcublas
LIBRARIES += -lcusparse
LIBRARIES += -lgomp

ifneq ($(MY_PATH),)
INCLUDES  += -I${MY_PATH}/build
LIBRARIES += -L${MY_PATH}/build -lL_API
endif

ifneq ($(NVCOMP_ROOT),)
INCLUDES  += -I$(NVCOMP_ROOT)/include
LIBRARIES += -L$(NVCOMP_ROOT)/lib -lnvcomp
ALL_CCFLAGS += -DZIPSERV_ENABLE_NVCOMP
endif
################################################################################

# Gencode arguments
SMS ?= 80 86 89
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++14
ALL_CCFLAGS += -maxrregcount=255 
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################
# Target rules
all: test_mm test_decompress bench_llama31_70b_compress bench_zipserv_gemm_sweep bench_llama31_70b_zipserv_vs_bitcomp_compress bench_kv_cache_zipserv_compress bench_kv_cache_zipserv_compress_scaling

test_mm.o:  test_mm.cu utils.h 
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

test_mm: test_mm.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

test_decompress.o:  test_decompress.cu utils.h 
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

test_decompress: test_decompress.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bench_llama31_70b_compress.o: bench_llama31_70b_compress.cu utils.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bench_llama31_70b_compress: bench_llama31_70b_compress.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bench_zipserv_gemm_sweep.o: bench_zipserv_gemm_sweep.cu utils.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bench_zipserv_gemm_sweep: bench_zipserv_gemm_sweep.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bench_llama31_70b_zipserv_vs_bitcomp_compress.o: bench_llama31_70b_zipserv_vs_bitcomp_compress.cu utils.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bench_llama31_70b_zipserv_vs_bitcomp_compress: bench_llama31_70b_zipserv_vs_bitcomp_compress.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bench_kv_cache_zipserv_compress.o: bench_kv_cache_zipserv_compress.cu utils.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bench_kv_cache_zipserv_compress: bench_kv_cache_zipserv_compress.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bench_kv_cache_zipserv_compress_scaling.o: bench_kv_cache_zipserv_compress_scaling.cu utils.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bench_kv_cache_zipserv_compress_scaling: bench_kv_cache_zipserv_compress_scaling.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f test_mm test_mm.o \
		test_decompress test_decompress.o \
		bench_llama31_70b_compress bench_llama31_70b_compress.o \
		bench_llama31_70b_codec bench_llama31_70b_codec.o \
		bench_zipserv_gemm_sweep bench_zipserv_gemm_sweep.o \
		bench_llama31_70b_zipserv_vs_bitcomp_compress bench_llama31_70b_zipserv_vs_bitcomp_compress.o \
		bench_kv_cache_zipserv_compress bench_kv_cache_zipserv_compress.o \
		bench_kv_cache_zipserv_compress_scaling bench_kv_cache_zipserv_compress_scaling.o

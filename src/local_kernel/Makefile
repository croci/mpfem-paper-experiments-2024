GCC_BASE_DIR=/home/mc2495/spack/opt/spack/linux-rocky8-sapphirerapids/gcc-14.1.0/gcc-14.2.0-4lgh7lyvifw3oamqy6tebb3ft6rbtj66
GCC_INSTALL_DIR=${GCC_BASE_DIR}/lib/gcc/x86_64-pc-linux-gnu/14.2.0
GCC_LIB_DIR=${GCC_BASE_DIR}/lib64

DEFAULT_COMPILER=g++
CLANG_FLAGS=-Wl,-rpath=${GCC_LIB_DIR} --gcc-install-dir=${GCC_INSTALL_DIR}

ifeq ($(shell echo ${DEFAULT_COMPILER} | grep -c 'clang++\|icpx'), 1)
DEFAULT_COMPILER_FLAGS=${CLANG_FLAGS}
else
DEFAULT_COMPILER_FLAGS=
endif

ifndef KERNEL_HEADER_FILE
	ifndef FLUSH_SUBNORMALS
		KERNEL_HEADER_FLAGS=
	else
		KERNEL_HEADER_FLAGS=-D FLUSH_SUBNORMALS=${FLUSH_SUBNORMALS}
	endif
else
	ifndef FLUSH_SUBNORMALS
		KERNEL_HEADER_FLAGS=-D KERNEL_HEADER=${KERNEL_HEADER_FILE}
	else
		KERNEL_HEADER_FLAGS=-D FLUSH_SUBNORMALS=${FLUSH_SUBNORMALS} -D KERNEL_HEADER=${KERNEL_HEADER_FILE}
	endif
endif

ifndef EXECUTABLE_FILENAME
	EXECUTABLE_FILENAME=kernel.run
endif

default: all

all: kernel

debug:
	$(DEFAULT_COMPILER) -march=sapphirerapids -O0 -g -fsanitize=address -std=c++23 ${DEFAULT_COMPILER_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

debug_oneapi:
	icpx -march=sapphirerapids -O0 -g -fsanitize=address -std=c++23 ${CLANG_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

debug_gcc:
	g++ -march=sapphirerapids -O0 -g -fsanitize=address -std=c++23 ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

debug_llvm:
	clang++ -march=sapphirerapids -O0 -g -fsanitize=address -std=c++23 ${CLANG_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

fast:
	$(DEFAULT_COMPILER) -march=sapphirerapids -Ofast -std=c++23 -mprefer-vector-width=512 ${DEFAULT_COMPILER_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

kernel:
	$(DEFAULT_COMPILER) -march=sapphirerapids -O3 -std=c++23 -mprefer-vector-width=512 ${DEFAULT_COMPILER_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

oneapi:
	icpx -march=sapphirerapids -O3 -std=c++23 -mprefer-vector-width=512 ${CLANG_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

gcc:
	g++ -march=sapphirerapids -O3 -std=c++23 -mprefer-vector-width=512 ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

llvm:
	clang++ -march=sapphirerapids -O3 -std=c++23 -mprefer-vector-width=512 ${CLANG_FLAGS} ${KERNEL_HEADER_FLAGS} -o ${EXECUTABLE_FILENAME} kernel.cpp

clean:
	rm *.run

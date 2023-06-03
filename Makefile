
all: fft-gpu

fft-gpu:
	clang \
		-Wall -Wextra -fdiagnostics-color=always \
		-O0 -g3 \
		\
		-I ../directOpenCL/include -I ../directOpenCL/cllib/include \
		-I include \
		-L ../directOpenCL/build \
		\
		src/fft-gpu.c \
		-lcl -lOpenCL



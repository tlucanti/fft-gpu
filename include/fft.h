
#ifndef FFT_H
#define FFT_H

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>

#include <barrier.h>

#define swap(a, b) do {		\
	typeof(a) c = a;	\
	a = b;			\
	b = c;			\
} while (false)

typedef double FLOAT;
typedef double complex COMPLEX;

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

void fft_parallel(COMPLEX *x, unsigned N);
void fft_fast(COMPLEX *x, unsigned int N);

#endif /* FFT_H */

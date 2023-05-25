
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

typedef double FLOAT;
typedef double complex COMPLEX;

#define PI 3.14159265358979323846264338328L
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define swap(a, b) do {		\
	typeof(a) c = a;	\
	a = b;			\
	b = c;			\
} while (false)

#define COMPLEX_SET(real, imag) ((real) + 1.0j * (imag))

#endif /* FFT_H */


#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdlib.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))
double PI;
typedef double complex cplx;

void _fft(cplx buf[], cplx out[], int n, int step)
{
	if (step < n) {
		_fft(out, buf, n, step * 2);
		_fft(out + step, buf + step, n, step * 2);

		for (int i = 0; i < n; i += 2 * step) {
			cplx t = cexp(-I * PI * i / n) * out[i + step];
			buf[i / 2]     = out[i] + t;
			buf[(i + n)/2] = out[i] - t;
		}
	}
}

void fft(cplx buf[], int n)
{
	cplx out[n];
	for (int i = 0; i < n; i++) out[i] = buf[i];

	_fft(buf, out, n, 1);
}

unsigned int reverse_bits(unsigned int b)
{
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16));
		return b;
}

void fft_fast(cplx x[], unsigned int N)
{
	unsigned int k = N, n;
	double thetaT = 3.14159265358979323846264338328L / N;
	cplx phiT = cos(thetaT) - 1j * sin(thetaT), T;
	while (k > 1)
	{
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++)
		{
			for (unsigned int a = l; a < N; a += n)
			{
				unsigned int b = a + k;
				cplx t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(N);
	for (unsigned int a = 0; a < N; a++)
	{
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			cplx t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
	//// Normalize (This section make it not working correctly)
	//Complex f = 1.0 / sqrt(N);
	//for (unsigned int i = 0; i < N; i++)
	//	x[i] *= f;
}

void show(const char * s, cplx buf[], int size) {
	printf("%s", s);
	for (int i = 0; i < size; i++) {
		if (i % 8 == 0) {
			printf("\n");
		}
		if (!cimag(buf[i]))
			printf("%g ", creal(buf[i]));
		else
			printf("(%g, %g) ", creal(buf[i]), cimag(buf[i]));
	}
}

void diff(cplx b1[], cplx b2[], int size) {
	for (int i = 0; i < size; ++i) {
		printf("(%.3g %.3g) (%.3g %.3g)\n",
			creal(b1[i]), cimag(b1[i]),
			creal(b2[i]), cimag(b2[i]));
	}
}

int main()
{
	PI = atan2(1, 1) * 4;
	cplx buf1[] = {1, 2, 3, 4, 5, 6, 7, 8};
	cplx buf2[] = {1, 2, 3, 4, 5, 6, 7, 8};

	fft(buf1, ARRAY_SIZE(buf1));
	fft_fast(buf2, ARRAY_SIZE(buf2));
	show("\nFFT : ", buf1, ARRAY_SIZE(buf1));
	show("\nFFT : ", buf2, ARRAY_SIZE(buf2));
	printf("\n");

	const int s = 4096;
	cplx b1[s];
	cplx b2[s];
	for (int i = 0; i < s; ++i) {
		b1[i] = rand();
		b2[i] = b1[i];
	}
	fft(b1, s);
	fft_fast(b2, s);
	//diff(b1, b2, s);
	for (int i = 0; i < s; ++i) {
		double d = cabs(b1[i] - b2[i]);
		if (d > cabs(b1[i] + b2[i]) / 100) {
			abort();
		}
	}
}

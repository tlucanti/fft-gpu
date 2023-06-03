
#include <fft.h>

#define PI 3.14159265358979323846264338328L
#define COMPLEX_SET(real, imag) ((real) + 1.0j * (imag))

volatile COMPLEX *g_x;
volatile COMPLEX *g_out;
volatile COMPLEX g_N;
struct barrier barrier;

static inline COMPLEX complex_mul(COMPLEX a, COMPLEX b)
{
	return a * b;
}

static inline bool is_in_range(int n, int start, int end, int step) {
	if (n >= start && n <= end) {
		return (n - start) % step == 0;
	} else {
		return false;
	}
}

static void *fft_parallel_worker(void *id)
{
	volatile COMPLEX *x = g_x;
	volatile COMPLEX *out = g_out;
	unsigned N = g_N;

	unsigned int pid = (unsigned long)id;
	unsigned int k = N;
	unsigned int n;
	unsigned int iter = 0;
	FLOAT thetaT = PI / N;
	COMPLEX phiT = COMPLEX_SET(cos(thetaT), -sin(thetaT));
	COMPLEX T;

	while (k > 1) {
		n = k;
		k >>= 1;
		phiT = complex_mul(phiT, phiT);
		T = 1.0;
		for (unsigned int l = 0; l < k; l++) {
			unsigned int a = pid;
			if (is_in_range(a, l, N, n)) {
				unsigned int b = a + k;
				COMPLEX t = x[a] - x[b];
				out[a] = x[a] + x[b];
				out[b] = t * T;
			}
			T = complex_mul(T, phiT);
		}
		barrier_wait(&barrier, pid, iter);
		//if (pid == 0) {
		//	printf("%d : ", k);
		//	show((void *)out, N);
		//}
		swap(out, x);
		++iter;
	}

	unsigned int m = (unsigned int)log2(N);
	unsigned int a = pid;
	unsigned int b = a;

	b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
	b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
	b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
	b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
	b = ((b >> 16) | (b << 16)) >> (32 - m);
	if (b > a) {
		COMPLEX t = x[a];
		x[a] = x[b];
		x[b] = t;
	}
	return NULL;
}

void fft_parallel(COMPLEX *x, unsigned N)
{
	pthread_t threads[N];
	COMPLEX out[N];

	g_x = x;
	g_out = out;
	g_N = N;
	barrier_init(&barrier, N);

	for (unsigned long i = 0; i < N; ++i) {
		if (pthread_create(&threads[i], NULL, fft_parallel_worker, (void *)i)) {
			printf("pthread_create error\n");
			exit(1);
		}
	}
	for (unsigned i = 0; i < N; ++i) {
		if (pthread_join(threads[i], NULL)) {
			printf("pthread_join error\n");
			exit(1);
		}
	}
}

void fft_fast(COMPLEX *x, unsigned int N)
{
	unsigned int k = N, n;
	FLOAT thetaT = PI / N;
	COMPLEX phiT = cos(thetaT) - 1.0j * sin(thetaT), T;
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
				COMPLEX t = x[a] - x[b];
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
			COMPLEX t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
}

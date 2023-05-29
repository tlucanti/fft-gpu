
#include <fft.h>
#include <barrier.h>

volatile COMPLEX *g_x;
volatile COMPLEX *g_out;
unsigned g_N;
struct barrier barrier;

#define __pr(x) (cabs(x) < 1e-3 ? 0.0 : x)

static void show(COMPLEX *buf, int size) {
	for (int i = 0; i < size; i++) {
		if (cabs(cimag(buf[i])) < 1e-3) {
			printf("%g ",
				__pr(creal(buf[i])));
		} else {
			printf("(%g, %g) ",
				__pr(creal(buf[i])),
				__pr(cimag(buf[i])));
		}
	}
	printf("\n");
}

static inline bool is_in_range(int n, int start, int end, int step) {
	if (n >= start && n <= end) {
		return (n - start) % step == 0;
	} else {
		return false;
	}
}

static inline COMPLEX complex_mul(COMPLEX a, COMPLEX b)
{
	return a * b;
}

void *_fft_parallel(void *id)
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

void fft_parallel(COMPLEX x[], unsigned N)
{
	printf("compute fft using %d threads\n", N);
	pthread_t threads[N];
	COMPLEX out[N];

	g_x = x;
	g_out = out;
	g_N = N;
	barrier_init(&barrier, N);

	for (unsigned long i = 0; i < N; ++i) {
		if (pthread_create(&threads[i], NULL, _fft_parallel, (void *)i)) {
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

void fft_fast(COMPLEX x[], unsigned int N)
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

void diff(COMPLEX b1[], COMPLEX b2[], int size) {
	for (int i = 0; i < size; ++i) {
		printf("(%.3g %.3g) (%.3g %.3g)\n",
			creal(b1[i]), cimag(b1[i]),
			creal(b2[i]), cimag(b2[i]));
	}
}

int main()
{
	const int size = 1 << 10;
	COMPLEX buf1[size];
	COMPLEX buf2[size];

	for (int i = 0; i < size; ++i) {
		buf1[i] = i + 1;
		buf2[i] = i + 1;
	}

	fft_parallel(buf1, ARRAY_SIZE(buf1));
	fft_fast(buf2, ARRAY_SIZE(buf2));
	printf("res : ");
	show(buf1, ARRAY_SIZE(buf1));
	//show(buf2, ARRAY_SIZE(buf2));
	printf("\n");

	return 0;
	const int s = 8;
	COMPLEX b1[s];
	COMPLEX b2[s];
	for (int i = 0; i < s; ++i) {
		b1[i] = rand();
		b2[i] = b1[i];
	}
	fft_parallel(b1, s);
	fft_fast(b2, s);
	//diff(b1, b2, s);
	for (int i = 0; i < s; ++i) {
		FLOAT d = cabs(b1[i] - b2[i]);
		if (d > cabs(b1[i] + b2[i]) / 100) {
			abort();
		}
	}
}

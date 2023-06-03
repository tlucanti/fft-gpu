
typedef float FLOAT;
typedef float2 COMPLEX;

#define PI 3.14159265358979323846264338328L
#define COMPLEX_SET(real, imag) (COMPLEX)(real, imag)

#define swap(a, b) do {		\
	__typeof(a) c = a;	\
	a = b;			\
	b = c;			\
} while (false)

static inline COMPLEX complex_mul(COMPLEX a, COMPLEX b)
{
	return COMPLEX_SET(
		a.x * b.x - a.y * b.y,
		a.x * b.y + a.y * b.x);
}

static inline bool is_in_range(int n, int start, int end, int step) {
	if (n >= start && n <= end) {
		return (n - start) % step == 0;
	} else {
		return false;
	}
}

__kernel void fft(__global COMPLEX *x, __global COMPLEX *buf, unsigned int N)
{
	unsigned int pid = get_global_id(0);
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
		T = COMPLEX_SET(1.0, 0);
		for (unsigned int l = 0; l < k; l++) {
			unsigned int a = pid;
			if (is_in_range(a, l, N, n)) {
				unsigned int b = a + k;
				COMPLEX t = x[a] - x[b];
				buf[a] = x[a] + x[b];
				buf[b] = complex_mul(t, T);
			}
			T = complex_mul(T, phiT);
		}
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		swap(buf, x);
		++iter;
	}

	unsigned int m = (unsigned int)log2((FLOAT)N);
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
}

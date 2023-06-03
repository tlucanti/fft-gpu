
#include <fft.h>
#include <barrier.h>

#define __pr(x) (fabs(x) < 1e-3 ? 0.0 : x)

static void show(COMPLEX *buf, int size) {
	for (int i = 0; i < size; i++) {
		if (fabs(cimag(buf[i])) < 1e-3) {
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

void diff(COMPLEX *b1, COMPLEX *b2, int size) {
	for (int i = 0; i < size; ++i) {
		FLOAT d = cabs(b1[i] - b2[i]) / cabs(b1[i] + b2[i]);
		if (d > 1e-4) {
			abort();
		}
	}
}

void fuzzer(const char *name, int size, int test_cnt)
{
	COMPLEX buf1[size];
	COMPLEX buf2[size];

	printf("%s:  0%%", name);
	fflush(stdout);
	for (int t = 0; t < test_cnt; ++t) {
		printf("\b\b\b%2d%%", t * 100 / test_cnt);
		fflush(stdout);
		for (int i = 0; i < size; ++i) {
			COMPLEX r = rand();
			buf1[i] = r;
			buf2[i] = r;
		}
		fft_parallel(buf1, size);
		fft_fast(buf2, size);
		diff(buf1, buf2, size);
	}
	printf("\b\b\bOK \n");
}

int main()
{
	fuzzer("fft small test", 1 << 4, 1000);
	fuzzer("fft medium test", 1 << 6, 100);
	fuzzer("fft large test", 1 << 10, 2);
}
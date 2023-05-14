
#include <iostream>
#include <complex>
#include <vector>
#include <cstdlib>

#include <fftw3.h>
#include <common.h>
#include <cllib/cllib.h>

void __fft_w3(size_t size, double *in, fftw_complex *out)
{
	auto plan = fftw_plan_dft_r2c_1d(size, in, out, FFTW_ESTIMATE);

	fftw_execute(plan);
	fftw_destroy_plan(plan);
}

void fft_w3(std::vector<double> &in,
	    std::vector<std::complex<double>> &out)
{
	panic_on(in.size() != out.size(), "in and out arrays size are not equal");
	__fft_w3(in.size(), in.data(), reinterpret_cast<fftw_complex *>(out.data()));
}


void __fft_cl(size_t size, double *in, cl_double2 *out)
{
        device_t device = create_device(gpu_type);
        context_t context = create_context(device);
        kernel_t kernel = create_kernel(device, context,
                        "#include <cl/fft.cl>\n#define V 1", "fft",
			"-I ../3rdparty/cllib");
        queue_t queue = create_queue(context, device);

        buffer_t bin = create_buffer(context, read_write, sizeof(double) * size);
        buffer_t bout = create_buffer(context, read_write, sizeof(cl_double2) * size);

        fill_buffer(queue, bin, in, true);

        set_kernel_arg_buffer(kernel, bin);
        set_kernel_arg_buffer(kernel, bout);

        run_kernel(queue, kernel, 10, 10);

        dump_buffer(queue, bout, out, true);
        flush_queue(queue);
}

void fft_cl(std::vector<double> &in,
	    std::vector<std::complex<double>> &out)
{
	panic_on(in.size() != out.size(), "in and out arrays size are not equal");
	__fft_cl(in.size(), in.data(), reinterpret_cast<cl_double2 *>(out.data()));
}

int main()
{
	std::vector<double> in = {1, 2, 3, 4, 5};
	std::vector<std::complex<double>> out(5);

	//fft_w3(in, out);
	fft_cl(in, out);

	for (auto x : out) {
		std::cout << x << ' ';
	}
	std::cout << '\n';
}
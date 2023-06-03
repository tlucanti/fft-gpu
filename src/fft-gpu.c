
#include <fft.h>
#include <cllib/cllib.h>

int main()
{
	COMPLEX x[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	unsigned int len = ARRAY_SIZE(x);
        unsigned int size = sizeof(x);

        device_t device = create_device(gpu_type);
        context_t context = create_context(device);
        kernel_t kernel = create_kernel(device, context,
                        "#include <src/fft.cl>\n#define V 1",
			"fft", "-I.");
        queue_t queue = create_queue(context, device);

        buffer_t x_mem = create_buffer(context, read_write, size);
        buffer_t buf_mem = create_buffer(context, read_write, size);

        fill_buffer(queue, x_mem, size, x, true);

        set_kernel_arg(kernel, x_mem);
        set_kernel_arg(kernel, buf_mem);
        set_kernel_arg(kernel, len);

        run_kernel(queue, kernel, 8, 1);

        dump_buffer(queue, x_mem, size, x, true);
        flush_queue(queue);

        for (unsigned int i = 0; i < len; ++i) {
                printf("(%g, %g) ", creal(x[i]), cimag(x[i]));
        }
        printf("\n");
}
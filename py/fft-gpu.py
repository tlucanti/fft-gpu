
import pyopencl as cl
import numpy as np

class FFTGPU():
	SOURCE_PATH = 'fft.cl'

	def __init__(self):
		self.context = cl.create_some_context()
		self.queue = cl.CommandQueue(self.context)
		source = self._read_source()
		self.kernel = cl.Program(self.context, source).build()

	def fft(self, array):
		cl_int = cl.cltypes.int
		cl_float2 = cl.cltypes.float2
		mf = cl.mem_flags
		size = len(array)

		x = np.array(array, dtype=cl_float2)
		buf = np.empty([size], dtype=cl_float2)

		x_mem = cl.Buffer(self.context,
				  mf.READ_WRITE | mf.COPY_HOST_PTR,
				  hostbuf=x)
		buf_mem = cl.Buffer(self.context,
				    mf.READ_WRITE | mf.COPY_HOST_PTR,
				    hostbuf=buf)

		self.kernel.fft(self.queue, [size], [1], x_mem, buf_mem, size)
		self.kernel.wait()

		cl.enqueue_copy(self.queue, x, x_mem).wait()
		return x

	def _read_source(self):
		with open(self.SOURCE_PATH, 'r') as f:
			return f.read()


def test():
	x = [1, 2, 3, 4, 5, 6, 7, 8]
	fftgpu = FFTGPU()
	y = fftgpu.fft(x)
	print(x)
	print(list(y))

if __name__ == '__main__':
	test()

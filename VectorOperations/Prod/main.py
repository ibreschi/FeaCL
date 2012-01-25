#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy
import pdb

class CL:
	def __init__(self):
		import os
		os.environ['PYOPENCL_COMPILER_OUTPUT']="1"
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)

	def loadProgram(self, filename):
		#read in the OpenCL source file as a string
		f = open(filename, 'r')
		fstr = "".join(f.readlines())
		print fstr
		#create the program
		self.program = cl.Program(self.ctx, fstr).build()

	def popCorn(self):

		mf = cl.mem_flags

		#initialize client side (CPU) arrays

		self.mat = numpy.array(range(16), dtype=numpy.float32)*2.0
		self.vec = numpy.array(range(4), dtype=numpy.float32)*3.0
		self.res = numpy.array(range(4), dtype=numpy.float32)		

		#create OpenCL buffers
		self.mat_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.mat)
		self.vec_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vec)
		self.res_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.res.nbytes)


if __name__ == "__main__":
	example = CL()
	example.loadProgram("part1.cl")
	example.popCorn()
	example.execute()


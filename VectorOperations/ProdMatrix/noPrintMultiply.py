
import pyopencl as cl
from time import time 
import numpy

#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT']="1"

class CL:
	def __init__(self):
		self.ctx = cl.create_some_context()
		for dev in self.ctx.devices:
			assert dev.local_mem_size >0
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

	def popCorn(self):
		block_size = 1
		self.a_width = 5 * block_size
		a_height = 10 * block_size
		b_width = 5 * block_size
		b_height = self.a_width
		c_width = b_width
		c_height = a_height
		self.h_a = numpy.random.rand(a_height, self.a_width).astype(numpy.float32)
		self.h_b = numpy.random.rand(b_height, b_width).astype(numpy.float32)
		self.h_c = numpy.empty((c_height, c_width)).astype(numpy.float32)

	
		print self.h_a.shape
		print self.h_b.shape
		print self.h_c.shape
		# cheking matrix dimension A = mxn B = nxk C =m*k
		assert self.h_a.shape[0] ==self.h_c.shape[0] and self.h_b.shape[1]==self.h_c.shape[1]
		#cheking data sizes dimension shoudl be divisible by block_size
		assert self.a_width % block_size == 0
		assert a_height % block_size == 0
		assert b_width % block_size == 0

		# init kernel parameters
		self.kernel_params = {"block_size":block_size, "w_a":self.a_width,"h_a":a_height,"w_b":b_width}
		mf = cl.mem_flags
		
		# Data transfer host -> device -------------
		self.d_a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_a)
		self.d_b_buf = cl.Buffer(self.ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_b)
		self.d_c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.h_c.nbytes)

	def loadProgram(self, filename):
		#read in the OpenCL source file as a string
		f = open(filename, 'r')
		fstr = "".join(f.readlines())
		#create the program

		if "NVIDIA" in self.queue.device.vendor:
			options = "-cl-mad-enable -cl-fast-relaxed-math"
		else:
			options = ""
		self.program = cl.Program(self.ctx, fstr % self.kernel_params,).build(options=options)
	
	def execute(self):
		kernel = self.program.matrixMul
		self.event = kernel(self.queue,self.h_c.shape,None,self.d_c_buf,self.d_a_buf,self.d_b_buf)
		cl.enqueue_copy(self.queue, self.h_c, self.d_c_buf)
		print "a", self.h_a
		print "b", self.h_b
		print "ris", self.h_c

if __name__ == "__main__":
	example = CL()
	example.popCorn()
	example.loadProgram("mult.cl")
	example.execute()



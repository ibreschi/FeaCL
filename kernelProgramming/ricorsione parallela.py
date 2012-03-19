
import pyopencl as cl
from time import time 
import numpy



class CL:
	def __init__(self):
		self.ctx = cl.create_some_context()
		for dev in self.ctx.devices:
			assert dev.local_mem_size >0
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
	
	def loadProgram(self, filename):
		#read in the OpenCL source file as a string
		f = open(filename, 'r')
		fstr = "".join(f.readlines())
		#create the program

		if "NVIDIA" in self.queue.device.vendor:
			options = "-cl-mad-enable -cl-fast-relaxed-math"
		else:
			options = ""
		self.program = cl.Program(self.ctx, fstr,).build(options=options)
		#self.program = cl.Program(self.ctx,KERNEL_CODE%self.kernel_params).build(options=options)

	def popCorn(self):
		self.a_dim = 30 
		self.h_a = numpy.empty(self.a_dim).astype(numpy.int32)
		for i in range (self.a_dim):
			self.h_a[i]=i

		self.h_c = numpy.empty(self.a_dim).astype(numpy.int32)
		mf = cl.mem_flags
		self.d_a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_a)
		self.d_c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.h_c.nbytes)

	
	def execute(self):
		kernel = self.program.fact
		self.event = kernel(self.queue,[self.a_dim],None,self.d_a_buf,self.d_c_buf)
		self.event.wait()
		cl.enqueue_copy(self.queue, self.h_c, self.d_c_buf)
		print "a", self.h_a
		print "ris", self.h_c

if __name__ == "__main__":
	example = CL()
	example.popCorn()
	example.loadProgram("ricorsione.cl")
	example.execute()



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
		#self.program = cl.Program(self.ctx,KERNEL_CODE%self.kernel_params).build(options=options)

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

		#cheking work_item_sizes
		for dev in self.ctx.devices:
			print dev.max_work_group_size
			print "max_mem_alloc_size: ",dev.max_mem_alloc_size
			print "max_parameter_size: ",dev.max_parameter_size

		# init kernel parameters
		self.kernel_params = {"block_size":block_size, "w_a":self.a_width,"h_a":a_height,"w_b":b_width}
		mf = cl.mem_flags
		
		# Data transfer host -> device -------------
		t1 = time ()
		self.d_a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_a)
		self.d_b_buf = cl.Buffer(self.ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_b)
		self.d_c_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=self.h_c.nbytes)
		self.push_time = time() -t1

	
	def execute(self):
		kernel = self.program.matrixMul
		# warmup ------------------------------------
		for i in range(5):
			event = kernel(self.queue,self.h_c.shape, None,self.d_c_buf,self.d_a_buf,self.d_b_buf)
			print " A  ", event.command_execution_status
			print " B  ", event.command_queue
			print " C  ", event.command_type
			print " D  ", event.context
			print " E  ", event.reference_count
		event.wait()
		#benchmark ---------------------------------
		t1= time()
		count = 2
		for i in range(count):
			self.event = kernel(self.queue,self.h_c.shape,None,self.d_c_buf,self.d_a_buf,self.d_b_buf)
		self.event.wait()

		self.gpu_time = (time()-t1)/(count+0.0)
		cl.enqueue_copy(self.queue, self.h_c, self.d_c_buf)
		self.pull_time = time()-t1
		#print "a", self.h_a
		#print "b", self.h_b
		#print "ris", self.h_c

	def timeOutput(self):
		gpu_total_time = self.gpu_time+self.push_time+self.pull_time
		print "GPU push+compute+pull total [s]:", gpu_total_time
		print "GPU push [s]:", self.push_time
		print "GPU pull [s]:", self.pull_time
		print "GPU compute (host-timed) [s]:", self.gpu_time
		print "GPU compute (event-timed) [s]: ", (self.event.profile.end-self.event.profile.start)*1e-9

		gflop = self.h_c.size * (self.a_width * 2.) / (1000**3.)
		gflops = gflop / self.gpu_time

		print
		print "GFlops/s:", gflops

		# cpu comparison --------------------------------------------------------------
		t1 = time()
		h_c_cpu = numpy.dot(self.h_a,self.h_b)
		cpu_time = time()-t1

		print
		print "GPU==CPU:",numpy.allclose(self.h_c, h_c_cpu)
		print
		print "CPU time (s)", cpu_time
		print

		print "GPU speedup (with transfer): ", cpu_time/gpu_total_time
		print "GPU speedup (without transfer): ", cpu_time/self.gpu_time


if __name__ == "__main__":
	example = CL()
	example.popCorn()
	example.loadProgram("mult.cl")
	example.execute()
	example.timeOutput()


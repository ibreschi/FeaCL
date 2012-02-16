
import pyopencl as cl
import numpy
import os
os.environ['PYOPENCL_COMPILER_OUTPUT']="1"

class CL:
	def __init__(self):
		self.ctx = cl.create_some_context()
		for dev in self.ctx.devices:
			assert dev.local_mem_size >0
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

	def popCorn(self):
		self.global_id = numpy.empty((32)).astype(numpy.int32)
		self.work_dim = numpy.empty((32)).astype(numpy.int32)
		self.global_size = numpy.empty((32)).astype(numpy.int32)
		self.global_id = numpy.empty((32)).astype(numpy.int32)
		self.local_size = numpy.empty((32)).astype(numpy.int32)
		self.local_id = numpy.empty((32)).astype(numpy.int32)
		self.num_groups = numpy.empty((32)).astype(numpy.int32)
		self.group_id = numpy.empty((32)).astype(numpy.int32)
		self.global_offset = numpy.empty((32)).astype(numpy.int32)
		
				
		mf = cl.mem_flags
		self.global_idBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.global_id)
		self.work_dimBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.work_dim)
		self.global_sizeBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.global_size)
		self.global_idBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.global_id)
		self.local_sizeBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.local_size)
		self.local_idBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.local_id)
		self.num_groupsBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.num_groups)
		self.group_idBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.group_id)
		self.global_offsetBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.global_offset)




	def loadProgram(self, filename):
		f = open(filename, 'r')
		fstr = "".join(f.readlines())

		if "NVIDIA" in self.queue.device.vendor:
			options = "-cl-mad-enable -cl-fast-relaxed-math"
		else:
			options = ""
		self.program = cl.Program(self.ctx, fstr,).build(options=options)
	
	def execute(self):
		global_work_size = (32,)

		for dev in self.ctx.devices:
			assert  dev.max_work_group_size >=  global_work_size [0]
			#print dev.max_compute_units


		kernel = self.program.ret
		print global_work_size
		local_size = (8,)
		print local_size
		#local_size = None

		self.event = kernel(self.queue,global_work_size,local_size,
			self.global_idBuffer ,
			self.work_dimBuffer ,
			self.global_sizeBuffer ,
			self.local_sizeBuffer ,
			self.local_idBuffer ,
			self.num_groupsBuffer ,
			self.group_idBuffer ,
			self.global_offsetBuffer)



		cl.enqueue_copy(self.queue, self.global_id, self.global_idBuffer)
		print "Global id",self.global_id
		cl.enqueue_copy(self.queue, self.work_dim ,self.work_dimBuffer)
		print "Work dimesion",self.work_dim
		cl.enqueue_copy(self.queue, self.global_size ,self.global_sizeBuffer)
		print "Global size",self.global_size
		cl.enqueue_copy(self.queue, self.local_size ,self.local_sizeBuffer)
		print "Local size",self.local_size
		cl.enqueue_copy(self.queue, self.local_id ,self.local_idBuffer)
		print "Local id",self.local_id
		cl.enqueue_copy(self.queue, self.num_groups ,self.num_groupsBuffer)
		print "Number of group",self.num_groups
		cl.enqueue_copy(self.queue, self.group_id ,self.group_idBuffer)
		print "Group id",self.group_id
		cl.enqueue_copy(self.queue, self.global_offset , self.global_offsetBuffer)
		print "Global offset",self.global_offset



if __name__ == "__main__":
	example = CL()
	example.popCorn()
	example.loadProgram("proveKernel.cl")
	example.execute()




import pyopencl as cl
import numpy

debug = 0

#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT']="1"

class CL:
	def __init__(self):
		self.ctx = cl.create_some_context()
		for dev in self.ctx.devices:
			assert dev.local_mem_size >0
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
		if (debug==1):
			print self.queue
			print "queue properties",self.queue.properties


	def popCorn(self):
		block_size = 8
		self.inputSignalWidth =  (numpy.int32(1*block_size))
		inputSignalHeight =  (numpy.int32(1*block_size))
		self.inputSignal = numpy.random.rand(inputSignalHeight, self.inputSignalWidth).astype(numpy.float32)*10
		self.maskWidth= (numpy.int32(3))
		self.mask = numpy.ones((self.maskWidth,self.maskWidth)).astype(numpy.float32)
		self.mask[1,1] =0.0

		self.outputSignalWidth =6
		self.outputSignalHeight =6
		self.outputSignal = numpy.empty((self.outputSignalWidth,self.outputSignalHeight)).astype(numpy.float32)

		mf = cl.mem_flags
		# Data transfer host -> device -------------
		self.inputSignalBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.inputSignal)
		self.maskBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.mask)
		self.outputSignalBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.outputSignal)


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
	
	def execute(self):
		global_work_size = [self.outputSignalWidth*self.outputSignalHeight]
		local_work_size = [1]
		if (debug==1):
			print global_work_size
			print local_work_size 
		kernel = self.program.convolve
		

		if (debug==1):
			print kernel.context
			print kernel.function_name
			print kernel.num_args
			print kernel.program
			print kernel.reference_count

		# Vecchia modalita' di creare un evento
		kernel.set_arg(0,self.inputSignalBuffer)
		kernel.set_arg(1,self.maskBuffer)
		kernel.set_arg(2,self.outputSignalBuffer)
		kernel.set_arg(3,self.inputSignalWidth)
		kernel.set_arg(4,self.maskWidth)
		self.event =cl.enqueue_nd_range_kernel(self.queue, kernel, global_work_size, local_work_size,
			 global_work_offset=None, wait_for=None, g_times_l=True)

		if (debug==1):
			wgi = cl.kernel_work_group_info
			for dev in self.ctx.devices:
				print "-------",dev,"-------" 
				print kernel.get_work_group_info(wgi.WORK_GROUP_SIZE,dev )
				print kernel.get_work_group_info(wgi.COMPILE_WORK_GROUP_SIZE,dev )
				print kernel.get_work_group_info(wgi.LOCAL_MEM_SIZE,dev )
				print kernel.get_work_group_info(wgi.PRIVATE_MEM_SIZE,dev )
				print kernel.get_work_group_info(wgi.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,dev )


		# Nuova modalita' di creare un evento
		self.event = kernel(self.queue,global_work_size,None,
			self.inputSignalBuffer,self.maskBuffer,self.outputSignalBuffer ,self.inputSignalWidth ,self.maskWidth)
		
		if (debug==1):
			print "context",self.event.context
			print "command_execution_status",self.event.command_execution_status
			print "command_queue",self.event.command_queue
			print "command_type",self.event.command_type
			print "reference_count",self.event.reference_count
			#print self.event.profile.end
			#print self.event.profile.queued
			#print self.event.profile.start
			#print self.event.profile.submit

		
		cl.enqueue_copy(self.queue, self.outputSignal, self.outputSignalBuffer)
		print self.inputSignal
		print self.mask

		print self.outputSignal

if __name__ == "__main__":
	example = CL()
	example.popCorn()
	example.loadProgram("convolve.cl")
	example.execute()



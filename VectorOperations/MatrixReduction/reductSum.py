
import pyopencl as cl
import numpy


class CL:
	def __init__(self):
		self.ctx = cl.create_some_context()
		for dev in self.ctx.devices:
			assert dev.local_mem_size >0
		self.queue = cl.CommandQueue(self.ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

	def popCorn(self):
		block_size = 8
		self.inputSignalWidth =  (numpy.int32(1*block_size))
		inputSignalHeight =  (numpy.int32(1*block_size))
		self.inputSignal = numpy.random.rand(inputSignalHeight, self.inputSignalWidth).astype(numpy.float32)*10
		rid = 3
		self.maskWidth= (numpy.int32(rid))

		self.outputSignalWidth =6
		self.outputSignalHeight =6
		self.outputSignal = numpy.empty((self.outputSignalWidth,self.outputSignalHeight)).astype(numpy.float32)

		mf = cl.mem_flags
		# Data transfer host -> device -------------
		self.inputSignalBuffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.inputSignal)
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
		#local_work_size = [1]
		kernel = self.program.nxnsum

		self.event = kernel(self.queue,global_work_size,None,
			self.inputSignalBuffer,self.outputSignalBuffer ,self.inputSignalWidth ,self.maskWidth)


		
		cl.enqueue_copy(self.queue, self.outputSignal, self.outputSignalBuffer)
		print self.inputSignal

		print self.outputSignal

if __name__ == "__main__":
	example = CL()
	example.popCorn()
	example.loadProgram("nxnSum.cl")
	example.execute()



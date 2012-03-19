
import pyopencl as cl
import numpy
import os
os.environ['PYOPENCL_COMPILER_OUTPUT']="1"


KERNEL_CODE = """  __kernel void ret( __global  int *  output1, __global  int *  output2){ 
	const int x = get_global_id(0);
	output1[x] = x ;
	barrier(CLK_LOCAL_MEM_FENCE);
	output2[x] = 1;
	}"""

plat = cl.get_platforms()
IntelCore2 = plat[0].get_devices()[0]
GF9400 = plat[0].get_devices()[1]
GF9600 = plat[0].get_devices()[2]
#ctx = cl.Context(devices = [IntelCore2])
ctx = cl.Context(devices = [GF9400])
#ctx = cl.Context(devices = [GF9600])
print ctx

queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

global_id = numpy.empty((32)).astype(numpy.int32)	
proof = numpy.empty((32)).astype(numpy.int32)						
mf = cl.mem_flags
global_idBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_id)
proofBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=proof)

if "NVIDIA" in queue.device.vendor:
	options = "-cl-mad-enable -cl-fast-relaxed-math"
else:
	options = ""
program = cl.Program(ctx, KERNEL_CODE,).build(options=options)
	
global_work_size = (32,)

for dev in ctx.devices:
	assert  dev.max_work_group_size >=  global_work_size [0]
			#print dev.max_compute_units


kernel = program.ret
print "Global Work Size ", global_work_size
local_size = (32,)
print "Local size ",local_size
print "Number of work group",(global_work_size[0] /local_size[0]) 

event = kernel(queue,global_work_size,local_size,global_idBuffer,proofBuffer )

cl.enqueue_copy(queue,global_id,global_idBuffer)
print global_id
cl.enqueue_copy(queue,proof,proofBuffer)
print proof



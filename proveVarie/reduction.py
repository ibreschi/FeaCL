import pyopencl as cl
from pyopencl.reduction import ReductionKernel
import numpy
from time import time 
import os
os.environ['PYOPENCL_COMPILER_OUTPUT']="1"
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)



dot = ReductionKernel (ctx, dtype_out=numpy.float32, neutral="0",
			reduce_expr="a+b" , map_expr="x[i]*y[i]" ,
			arguments="__global const float *x, __global const float *y")

import pyopencl.clrandom as cl_rand

x= cl_rand.rand(queue,(1000*1000),dtype=numpy.float32)
y= cl_rand.rand(queue,(1000*1000),dtype=numpy.float32)

t1= time()
x_dot_y = dot(x,y).get()
gpu_time = (time()-t1)



t1 = time()
x_dot_y_cpu = numpy.dot(x.get(),y.get())
cpu_time = time()-t1



print "CPU time (s)", cpu_time

print "compute (host-timed) [s]", gpu_time
import numpy 
import pyopencl as cl
import pyopencl.array as cl_array

from time import time 

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
randArr=numpy.random.randn(4,4).astype(numpy.float32)


a_gpu = cl_array.to_device(queue ,randArr)
t1= time()
a_doubled = (20000*a_gpu).get()
#2*a_gpu
gpu_time = (time()-t1)

print "tempo assegnazione e moltiplicazione con gpu",gpu_time


a_cpu = randArr
t2 = time()
a_doubledCpu = 20000*a_cpu
#2*a_cpu
cpu_time =(time()-t2)

print "tempo assegnazione e moltiplicazione con cpu",cpu_time



n=10000
a_gpu = cl_array.to_device(queue, numpy.random.randn(n).astype(numpy.float32))
b_gpu = cl_array.to_device(queue, numpy.random.randn(n).astype(numpy.float32))

from pyopencl.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(ctx,
			"float a, float *x, float b, float *y, float *z ",
				"z[i]=a*x[i]+b*y[i]")


c_gpu= cl_array.empty_like(a_gpu)
t3 =time()
lin_comb(5,a_gpu,6,b_gpu,c_gpu)
gpu_timeLinComb = (time()-t3)
#print gpu_timeLinComb

import numpy.linalg as la 
assert la.norm((c_gpu-(5*a_gpu+6*b_gpu)).get()) <1e-5

n= 100
prod = ElementwiseKernel(ctx,
			"float *x, float *y ,float *z",
			"z[i]=x[i]*y[i]")

c = numpy.random.randn(n).astype(numpy.float32)
d = numpy.random.randn(n).astype(numpy.float32)
c_gpu = cl_array.to_device(queue,c )
d_gpu = cl_array.to_device(queue,d )
e_gpu= cl_array.empty_like(c_gpu)
t4 =time()
prod(c_gpu,d_gpu,e_gpu)
gpu_timeProd = (time()-t4)

t5 =time()
risCpu = c_gpu*d_gpu 
cpu_timeProd = (time()-t4)

print la.norm((e_gpu-risCpu).get())
print gpu_timeProd
print cpu_timeProd






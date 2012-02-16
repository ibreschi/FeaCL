import numpy 
import pyopencl as cl
from pyopencl.scan import InclusiveScanKernel
import pyopencl.array as cl_array
##
##
# Somma di vettore ordinato n interi random tra 0 e 9  
##

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

knl = InclusiveScanKernel(ctx , numpy.int32, "a+b")

n= 2**20 -2**18+5

host_data = numpy.random.randint(0,10,n).astype(numpy.int32)
dev_data = cl_array.to_device(queue, host_data)

a =knl(dev_data)

b= numpy.cumsum(host_data, axis=0)
assert(dev_data.get() == numpy.cumsum(host_data, axis=0)).all()
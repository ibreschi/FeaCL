import time

class Timing(object):
    def __init__(self):
        self.timings = {}
        self.col = self.__collector()
        self.col.next()                 #coroutine syntax

    def __collector(self):
        while True:
            (name, t) = (yield)         #coroutine syntax
            if name in self.timings:
                self.timings[name]["timings"] += [t]
                self.timings[name]["count"] += 1
                self.timings[name]["total"] += t
            else:
                self.timings[name] = {} #if this entry doesn't exist yet
                self.timings[name]["timings"] = [t]
                self.timings[name]["count"] = 1
                self.timings[name]["total"] = t

    def __call__(self, func):
        """Turn the object into a decorator"""
        def wrapper(*arg, **kwargs):
            t1 = time.time()                #start time
            res = func(*arg, **kwargs)      #call the originating function
            t2 = time.time()                #stop time
            t = (t2-t1)*1000.0              #time in milliseconds
            data = (func.__name__, t)
            self.col.send(data)             #collect the data
            return res 
        return wrapper


    def __str__(self):
        s = "Timings:\n"
        #print dir(self)
        for key in self.timings.keys():
            s += "%s | " % key 
            count = self.timings[key]["count"]
            total = self.timings[key]["total"]
            s += "average: %s | total: %s | count: %s\n" % (total / count, total, count)
        return "%s" % s 



if __name__ == "__main__":


    timings = Timing()
    @timings
    def add(x,y):
        for i in range(10000):
            c = x + y
        return c

    @timings
    def multiply(x,y):
        for i in range(10000):
            c = x * y
        return c

    for i in range(100):
        add(3.,4.)
        multiply(3., 4.)

    print timings

    

    import pyopencl as cl
    from pyopencl.elementwise import ElementwiseKernel
    import pyopencl.array as cl_array
    import numpy

    t1= Timing()

    @t1 
    def work(x,y,n):
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        arr =cl_array.to_device(queue, numpy.zeros(n).astype(numpy.float16))
        ris = cl_array.to_device(queue,numpy.zeros(1).astype(numpy.float32))
        summ = ElementwiseKernel(ctx,
            "float a,float b, float *x,float *c ",
            "c[0]=a+b ") 
        prod = ElementwiseKernel(ctx,
            "float a,float b, float *x, float *c ",
            "c[0]=a*b ")
        summ (x,y,arr,ris)
        prod(x,y,arr,ris)   
        return ris

    print work(3.0,4.0,1800000)
    
    print t1



###################
## This is a simple PyOpenCL example
##
## The purpose is to evaluate Bicubic Bezier surfaces on the GPU, 
## AUTHOR: Shayan Javed
##

import sys
import pyopencl as cl
import numpy
import time
#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT']="1"


# Global variables
degreeU = 0
degreeV = 0
detail = 0.0

# the function to read a bezier file - returns list of vertices
def readBezierFile(fileName):
    # open the file first
    file = open(fileName)
    
    # read all the text
    str = file.read()
    
    # split it up line-by-line
    lines = str.split('\n')
    
    # first line - degree + detail
    line1 = lines[0].split()
    global degreeU
    global degreeV
    global detail
    degreeU = int(line1[0])
    degreeV = int(line1[1])
    detail = float(line1[2])
    order = int(degreeU) + 1
    
    # create list
    # numpy array - make sure to specify astype as float32 to avoid /128 errors 
    vertices = numpy.empty( (order * order, 4)).astype(numpy.float32)
    # go through the rest of the lines, read in the curve
    index = 1
    
    while index <= (len(lines) - 1):
        # read line
        line = lines[index].split()
       
        # create a new list
        newList = [line[0], line[1], line[2], 0]
        
         # assign values
        vertices[index-1] = newList
        
        # increment
        index = index + 1
    
    return vertices

# reads a file and returns all the text as a string
def readFile(fileName):
    # open the file first
    file = open(fileName, 'r')
    
    # read all the text
    str = "".join(file.readlines())
    
    return str

# the main function
def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    # try to read from a file here - returns the array
    vertices = readBezierFile("patch1")
    #print vertices
    
    #print "Vertices:" 
    #print vertices.size
    
    # read the kernel file
    kernelString = readFile("bezier2.cl")
    
    # the bezier curve TODO: fix to surface instead
    #curve1 = BezierCurve(len(vertices), 0.1, vertices)
    
    # Array of UV values - 36 in total (detail by 0.2)
    uvValues = numpy.empty( (36,2 )).astype(numpy.float32)
    index = 0
    for u in range(0, 12, 2): # step = 2
        for v in range(0, 12, 2):
            # conver the ints to floats
            fU = float(u)
            fV = float(v)
            uvValues[index] = [ fU/10.0, fV/10.0]
            index = index + 1
         

    print "\nUV Values: "
    print uvValues
    print uvValues.size
   
    ##########################
    ### OPENCL SETUP
    #########################

    plat = cl.get_platforms()
    GF9400 = plat[0].get_devices()[1]
    ctx = cl.Context(devices = [GF9400])
    cq = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags
    # the control point vertices
    print vertices.size
    vertex_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
    # the uv buffer
    uv_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uvValues)
    # final output
    output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, uvValues.nbytes * 2) # double the amount of value
    
    if "NVIDIA" in cq.device.vendor:
        options = "-cl-mad-enable -cl-fast-relaxed-math"

    else:
        options = ""
    prg = cl.Program(ctx, kernelString).build(options=options)
    
    # the global size (number of uv-values): 36 * 9 work-items (One Work-group per UV value) 
    globalSize = (36*9 ,)
    # local size
    localSize = (9,) # for bicubic patches

    # work-groups tutti di dimensioni uguali
    for i in range(len(globalSize)):
        assert (globalSize[i] % localSize[i] ) == 0
    # controlli sui size
    for dev in ctx.devices:
        ris =0
        for i in range(len(globalSize)):
            ris+= globalSize[i]
        assert ris < dev.max_work_group_size

    # timer start
    start = time.time()

    local_points =  cl.LocalMemory(9 * numpy.dtype('float32').itemsize * 4)
    local_points2 =  cl.LocalMemory(4 * numpy.dtype('float32').itemsize * 4)

    exec_evt = prg.bezierEval(cq, globalSize, localSize, vertex_buffer, uv_buffer, output_buffer,local_points,local_points2)
    exec_evt.wait()
    
    # elapsed time
    elapsed = exec_evt.profile.end - exec_evt.profile.start
    print("Execution time of test: %g " % elapsed)
    eval = numpy.empty( 36*4).astype(numpy.float32)
    cl.enqueue_copy(cq, eval ,output_buffer )
    print "It took: ", time.time() - start,"seconds."
    
    # Output to a file
    f = open('output', 'w')
    index = 0
    j = 6
    while index < 36:
        val1 = eval[index * 4]
        val2 = eval[index * 4 + 1]
        val3 = eval[index * 4 + 2]
        val4 = eval[index * 4 + 3]
        f.write(repr(round(val1, 2)).rjust(j))
        f.write(" ")
        f.write(repr(round(val2, 2)).rjust(j))
        f.write(" ")
        f.write(repr(round(val3, 2)).rjust(j))
        f.write(" ")
        f.write(repr(round(val4, 2)).rjust(j))
        f.write("\n")
        
        index = index + 1
    
    #print eval
    
# main invocation
if __name__ == "__main__":
    sys.exit(main())
from OpenGL.GL import *
from OpenGL.GLU import *

import pyopencl as cl
import numpy

# Global variables
degreeU = 0
degreeV = 0
detail = 0.0

class OpenClOperations(object):
    def __init__(self, *args, **kwargs):
        self.clinit()
        self.loadProgram("bezier2.cl");
        self.popCorn()

    def clinit(self):
        from pyopencl.tools import get_gl_sharing_context_properties
        plat = cl.get_platforms()
        GF9400 = plat[0].get_devices()[1]
        self.ctx = cl.Context(devices = [GF9400],properties=get_gl_sharing_context_properties())
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


    def initBuffers(self):
        self.num= len(self.result)
        self.vertices = numpy.ndarray((self.num/4, 4), dtype=numpy.float32)
        for i in range(self.num):
            self.vertices[i/4,i%4] = self.result[i] 
        from OpenGL.arrays import ArrayDatatype as ADT
    
        self.bufferVertices = GLuint(0)
        self.bufferVertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER_ARB, self.bufferVertices)
        glBufferData(GL_ARRAY_BUFFER_ARB, ADT.arrayByteCount(self.vertices), 
            ADT.voidDataPointer(self.vertices), GL_STATIC_DRAW_ARB)


    def loadProgram(self,fileName):
        # open the file first
        file = open(fileName, 'r')
    
        # read all the text
        kernelString = "".join(file.readlines())

        if "NVIDIA" in self.queue.device.vendor:
            options = "-cl-mad-enable -cl-fast-relaxed-math"

        else:
            options = ""
        self.program = cl.Program(self.ctx, kernelString).build(options=options)
  

    # the function to read a bezier file - returns list of vertices
    def readBezierFile(self,fileName):
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


    def writeOutput(self,result):
        # Output to a file
        f = open('output', 'w')
        index = 0
        j = 6
        while index < 36:
            val1 = result[index * 4]
            val2 = result[index * 4 + 1]
            val3 = result[index * 4 + 2]
            val4 = result[index * 4 + 3]
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
    


    def render(self):        
        glEnable(GL_POINT_SMOOTH)
        glPointSize(3)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER_ARB,self.bufferVertices)
        glVertexPointer(3,GL_FLOAT,0,None)      
        glDrawArrays(GL_POINTS,0,len(self.vertices))
        glDisableClientState(GL_VERTEX_ARRAY)
        
        glDisable(GL_BLEND)
        

     

    def popCorn(self):
        vertices = self.readBezierFile("patch1")
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

        mf = cl.mem_flags
        vertex_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
        # the uv buffer
        uv_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uvValues)
        # final output
        output_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, uvValues.nbytes * 2)


        # the global size (number of uv-values): 36 * 9 work-items (One Work-group per UV value) 
        globalSize = (36*9,)
        # local size
        localSize = (9,) # for bicubic patches
        # work-groups tutti di dimensioni uguali
        for i in range(len(globalSize)):
            assert (globalSize[i] % localSize[i] ) == 0
        # controlli sui size
        for dev in self.ctx.devices:
            ris =0
            for i in range(len(globalSize)):
                ris+= globalSize[i]
            assert ris < dev.max_work_group_size

        # timer start

        local_points =  cl.LocalMemory(9 * numpy.dtype('float32').itemsize * 4)
        local_points2 =  cl.LocalMemory(4 * numpy.dtype('float32').itemsize * 4)

        exec_evt = self.program.bezierEval(self.queue, globalSize, localSize, vertex_buffer, uv_buffer, output_buffer,local_points,local_points2)
        exec_evt.wait()

         # elapsed time
        elapsed = exec_evt.profile.end - exec_evt.profile.start
        print("Execution time of test: %g " % elapsed)
        self.result = numpy.empty( 36*4).astype(numpy.float32)
        cl.enqueue_copy(self.queue, self.result ,output_buffer )        

        self.initBuffers()
        self.render()

        #self.writeOutput(self.result)

    


    

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 12:09:48 2011

@author: omar.abo-namous
"""
"""
Copyright (c) 2011 Omar Abo-Namous

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
"""

__author__ = "Omar Abo-Namous"
__copyright__ = "Copyright 2011"
__credits__ = ["Ian Johnson","Andreas Klöckner"]
__license__ = "MIT License (MIT)"
__version__ = "0.1"
__maintainer__ = "Omar Abo-Namous"
__email__ = "kontakt@toomuchcookies.net"
__status__ = "Development"

from OpenGL import GL as gl, GLU as glu, GLUT as glut
from OpenGL.arrays import vbo
import numpy as np
import pyopencl as cl
import scipy
import os, sys

class test(object):
    width = 800
    height = 600
    
    def __init__(self):
        self.makeMovie = False
        self.mouse_down = False
        self.mouse_old = [0., 0.]
        self.rotate = [0., 0., 0.]
        self.translate = [0., 0., 0.]
        self.initrans = [0., 0., -2.]
        self.dt = np.float32(.001)
        self.size = 100
        size = self.size
        self.randarr = np.random.rand(self.size, self.size, self.size)
        self.randarr = np.require(self.randarr, 'f')
        arr = [(float(x)/(size-1)-.5,float(y)/(size-1)-.5,float(z)/(size-1)-.5,1.0) for x in xrange(size) for y in xrange(size) for z in xrange(size)]
        self.arr = np.require(arr, 'f')
        colarr = [(.5-float(x)/size+.2,float(y)/size-.5+.2,float(z)/size-.5+.2,1.0) for x in xrange(size) for y in xrange(size) for z in xrange(size)]
        self.colarr = np.require(colarr, 'f')
        if self.makeMovie:
            self.curimage = np.zeros((self.height, self.width, 3),dtype=np.uint8)
            self.curindex = 0

        glut.glutInit()
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
        glut.glutInitWindowSize(self.width, self.height)
        glut.glutInitWindowPosition(0, 0)
        self.win = glut.glutCreateWindow("Testing...")
        glut.glutDisplayFunc(self.draw)
        glut.glutKeyboardFunc(self.on_key)
        glut.glutMouseFunc(self.on_click)
        glut.glutMotionFunc(self.on_mouse_motion)
        glut.glutTimerFunc(10, self.timer, 10)
        self.glinit()
        self.clinit()
        self.loadProgram("kernel.cl")
        self.loadclData()
        #glut.glutMainLoop()
        
    def clinit(self):
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        if sys.platform == "darwin":
            self.ctx= cl.create_some_context()
            #print cl.context_properties.
            #print get_gl_sharing_context_properties()
            #print "ctx_props.CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE", cl.get_apple_cgl_share_group()
            print cl.get_apple_cgl_share_group()
            #self.ctx = cl.Context(properties=get_gl_sharing_context_properties(), devices=[])
        else:
            self.ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)
                
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        f = open(filename, 'r')
        fstr = f.read()
        self.program = cl.Program(self.ctx, fstr).build()

    def glinit(self):
        gl.glViewport(0, 0, self.width, self.height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(60., self.width / float(self.height), .1, 1000.)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        size = self.size
        self.arrvbo = vbo.VBO(data=self.arr, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.arrvbo.bind()
        self.colvbo = vbo.VBO(data=self.colarr, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.colvbo.bind()

    ###GL CALLBACKS
    def timer(self, t):
        glut.glutTimerFunc(t, self.timer, t)
        glut.glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            if self.makeMovie:
                os.system("rm movie.mp4")
                os.system("ffmpeg -r "+str(10)+" -i _tmp%05d.png -vcodec libx264 -vpre hq -vpre ipod640 -b 250k -bt 50k -acodec libfaac -ab 56k -ac 2 -s 480x320 movie.mp4")
                os.system("ffmpeg2theora movie.ogv movie.mp4")
                os.system("rm _tmp*.png")
            sys.exit()
        elif args[0] == 'm':
            print gl.glGetFloatv( gl.GL_MODELVIEW_MATRIX )

    def on_click(self, button, state, x, y):
        if state == glut.GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old[0] = x
        self.mouse_old[1] = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old[0]
        dy = y - self.mouse_old[1]
        if self.mouse_down and self.button == 0: #left button
            self.rotate[0] += dy * .2
            self.rotate[1] += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            self.translate[2] -= dy * .01
        self.mouse_old[0] = x
        self.mouse_old[1] = y
    ###END GL CALLBACKS
    
    def draw(self):
        self.execute()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        #handle mouse transformations
        gl.glTranslatef(self.initrans[0], self.initrans[1], self.initrans[2])
        gl.glRotatef(self.rotate[0], 1, 0, 0)
        gl.glRotatef(self.rotate[1], 0, 1, 0) #we switched around the axis so make this rotate_z
        gl.glTranslatef(self.translate[0], self.translate[1], self.translate[2])

        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glPointSize(5)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.arrvbo.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(4, gl.GL_FLOAT, 0, self.arrvbo)
        self.colvbo.bind()
        gl.glColorPointer(4, gl.GL_FLOAT, 0, self.colvbo)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.size**3)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        self.arrvbo.unbind()
        self.colvbo.unbind()
        glut.glutSwapBuffers()
        
        if self.makeMovie:
            gl.glReadPixels(0,0,800,600,gl.GL_RGB, gl.GL_UNSIGNED_BYTE, array=self.curimage)
            fname = '_tmp%05d.png'%self.curindex
            scipy.misc.imsave(fname, self.curimage)
            self.curindex += 1
    
    def loadclData(self):
        mf = cl.mem_flags
        print cl.have_gl()


        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.arrvbo.bind()
        self.arr_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.arrvbo.buffers[0]))
        self.colvbo.bind()
        self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.colvbo.buffers[0]))
        
        self.randarr_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.randarr)

        self.arrvbo.unbind()
        self.colvbo.unbind()
        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.arr_cl, self.col_cl]

    def execute(self):
        self.randarr = np.random.rand(self.size, self.size, self.size)
        self.randarr = np.require(self.randarr, 'f')
        cl.enqueue_copy(self.queue, self.randarr_cl, self.randarr)
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects).wait()
        global_size = (self.size**3,)
        local_size = None
        kernelargs = (self.arr_cl, 
                      self.col_cl,
                      self.randarr_cl,
                      self.dt)
        self.program.mykernel(self.queue, global_size, local_size, *(kernelargs))
        cl.enqueue_release_gl_objects(self.queue, self.gl_objects).wait()
        self.queue.finish()
        
if __name__ == '__main__':
    mytest = test()
    print "miao"

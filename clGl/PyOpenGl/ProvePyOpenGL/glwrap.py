from OpenGL import GL
from OpenGL.GL.ARB import vertex_array_object

# I'm not entirely sure why I have to look inside OpenGL.GL.ARB
# to find this function, nor why PyOpenGL doesn't provide its usual pythonic
# wrapper for it. I suspect it's because I only have OpenGL2.1 hardware,
# instead of 3.0 or greater.
def glGenVertexArray():
    '''
    Return the integer ID of the created vertex object array
    We always create one ID - we're not as flexible as the underlying OpenGL
    function which could create an array of several of them.
    '''
    vao_id = GL.GLuint(0)
    vertex_array_object.glGenVertexArrays(1, vao_id)
    return vao_id.value
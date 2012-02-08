__kernel void mykernel(__global float4* pos, __global float4* color, __global float* rand, float dt)
{
    unsigned int i = get_global_id(0);
    if (rand[i] <.05) { color[i].w = 0; }
    if (rand[i]> .999) { color[i].w = 1; }
}
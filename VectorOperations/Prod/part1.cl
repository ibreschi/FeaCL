__kernel void part1(__global float* a, __global float* b, __global float* c)
{
    unsigned int i = get_global_id(0);

    c[i] = a[i] + b[i];
}

__kernel void part2(__global float* a, __global float* b, __global float* c)
{
    unsigned int i = get_global_id(0);
    i=0;
}

__kernel void part3(__global float* matrix, __global float* vector, __global float* result)
{
    unsigned int i = get_global_id(0);
    result[i] = dot(as_float(matrix[i]) , as_float(vector[0]) );
    result[i] = vector[2]; 
}
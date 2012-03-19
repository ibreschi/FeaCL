
int factRic(int x){

	if (x==0)
		return 1;
	else 
		return (x*factRic(x-1));

}

__kernel void fact(__global int* a, __global int* c)
{
    unsigned int i = get_global_id(0);

    c[i] = factRic(a[i]);
}

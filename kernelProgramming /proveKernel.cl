

kernel void
my_func_a(global float *src, global float *dst, local float *l_var)
{
    
 }
kernel void
my_func_b(global float * src, global float *dst, local float *l_var)
{
    my_func_a(src, dst, l_var);
}

__kernel void ret( __global  int *  output1,__global  int *  output2 , 
	__global  int *  output3,__global  int *  output4 ,
		__global  int *  output5,__global  int *  output6, 
		__global  int *  output7,__global  int *  output8){
	
	const int x = get_global_id(0);
	output1[x] = x ;

	// ritorna la dimensione del work group 1d 2d 3d
	output2[x] = get_work_dim();
	// ritorna la dimensione del global size 1d con 0, 2d con 1, 3d con 2
	output3[x]=get_global_size(0);
	// ritorna l'id del work-item nello spazio 1d con 0 2d con 1 3d con 2
	output4[x] = get_local_size(0);
	output5[x]= get_local_id(0);
	output6[x] = get_num_groups(0);
	output7[x] = get_group_id(0);
	output8[x] = get_global_offset(0);

	//barrier(CLK_LOCAL_MEM_FENCE);

}

   
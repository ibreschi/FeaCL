
import pyopencl as cl
l = [ 1,2,3,4,5]

for i in range( 5):
	assert l[i]> 0



ctx = cl.create_some_context()

for device in ctx.devices:
	print "------------------------"
	print "Nome: ", device
	print "Bits di indirizzamento: ",device.address_bits
	print "Device disponibile: ",device.available
	print "Versione driver: ",device.driver_version
	print "Little endian: ",device.endian_little
	print "Global Cache size: ",device.global_mem_cache_size
	print "Global Cache type: ",device.global_mem_cache_type
	print "Global Mem size:",device.global_mem_size
	print "Local mem syze: ",device.local_mem_size
	print "Local mem type: ",device.local_mem_type
	print "Device clock frequ: ",device.max_clock_frequency
	print "Unita di calcolo: ",device.max_compute_units
	print "Nome ",device.name
	print "Profilo: ",device.profile
	print "Tipo: ",device.type
	print "Vendor: ",device.vendor
	print "Vendor_id: ",device.vendor_id
	print "Versione: ",device.version
	print "max_constant_args: ",device.max_constant_args
	print "max_constant_buffer_size: ",device.max_constant_buffer_size
	print "max_mem_alloc_size: ",device.max_mem_alloc_size
	print "max_parameter_size: ",device.max_parameter_size
	print "max_read_image_args: ",device.max_read_image_args
	print "max_samplers: ",device.max_samplers
	print "max_work_item_dimensions: ",device.max_work_item_dimensions
	print "max_work_item_sizes: ",device.max_work_item_sizes
	print "max_max_work_group_size",device.max_work_group_size
	print "max_write_image_args: ",device.max_write_image_args
	print "mem_base_addr_align: ",device.mem_base_addr_align
	print "min_data_type_align_size: ",device.min_data_type_align_size




	print "------------------------"


	import sys

print 'maxint    :', sys.maxint
print 'maxsize   :', sys.maxsize
print 'maxunicode:', sys.maxunicode

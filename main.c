#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
//#include <malloc.h>
//#include <string.h>
#include <alloca.h>
#include <CL/opencl.h>

#include "kernel.h"

#  define OCL_SAFE_CALL(call) {                                              \
    cl_int err = call;                                                       \
    if( CL_SUCCESS != err) {                                                 \
        fprintf(stderr, "OpenCL error in file '%s' in line %i : Code %d.\n", \
                __FILE__, __LINE__, err );                                   \
        exit(EXIT_FAILURE);                                                  \
    } }

unsigned int pow2(unsigned int v){
	return 1 << v;
}

void init_data(int *data, unsigned int len){
	for(int i=0; i<len; i++)
		data[i] = 0;
}

void flushed_printf(const char* format, ...){
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

void CL_CALLBACK ctxErrorCallback(const char *errInfo, const void *private_info, size_t cb, void *user_data){
	fprintf(stderr, "OpenCL Error (CB): %s\n", errInfo);
}

void cl_helper_PrintAvailableDevices(void){
	cl_uint cnt_platforms, cnt_device_ids;
	cl_platform_id *platform_ids;
	OCL_SAFE_CALL( clGetPlatformIDs(0, NULL, &cnt_platforms) );

	platform_ids = (cl_platform_id*)alloca(sizeof(cl_platform_id)*cnt_platforms);
	OCL_SAFE_CALL( clGetPlatformIDs(cnt_platforms, platform_ids, NULL) );

	printf("Available devices:\n");
	int cur_dev_idx = 1;
	for(int i=0; i<(int)cnt_platforms; i++){
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &cnt_device_ids) );

		size_t t;
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL, &t) );
		char *cl_plf_name = (char*)alloca( t );
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, t, cl_plf_name, NULL) );

/*		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, 0, NULL, &t) );
		char *cl_plf_vendor = (char*)alloca( t );
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, t, cl_plf_vendor, NULL) );*/

		cl_device_id *device_ids = (cl_device_id*)alloca(sizeof(cl_device_id)*cnt_device_ids);
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, cnt_device_ids, device_ids, NULL) );

		for(int d=0; d<(int)cnt_device_ids; d++){
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, 0, NULL, &t) );
			char *cl_dev_name = (char*)alloca( t );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, t, cl_dev_name, NULL) );

			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR, 0, NULL, &t) );
			char *cl_dev_vendor = (char*)alloca( t );
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_VENDOR, t, cl_dev_vendor, NULL) );
			
			cl_device_type dev_type;
			OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL) );
			const char *dev_type_str;
			switch(dev_type){
				case CL_DEVICE_TYPE_CPU:
					dev_type_str = "CPU";
					break;
				case CL_DEVICE_TYPE_GPU:
					dev_type_str = "GPU";
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					dev_type_str = "Accelerator";
					break;
				default:
					"Other";
			}

			printf(" %d. %s: %s [%s/%s]\n", cur_dev_idx, dev_type_str, cl_dev_name, cl_dev_vendor, cl_plf_name);

			cur_dev_idx++;
		}
	}
}

cl_device_id cl_helper_SelectDevice(int dev_idx){
	cl_device_id device_selected = (cl_device_id)-1;
	cl_uint cnt_platforms, cnt_device_ids;
	cl_platform_id *platform_ids;

	OCL_SAFE_CALL( clGetPlatformIDs(0, NULL, &cnt_platforms) );
	platform_ids = (cl_platform_id*)alloca(sizeof(cl_platform_id)*cnt_platforms);
	OCL_SAFE_CALL( clGetPlatformIDs(cnt_platforms, platform_ids, NULL) );

	int cur_dev_idx = 1;
	for(int i=0; i<(int)cnt_platforms; i++){
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &cnt_device_ids) );
		cl_device_id *device_ids = (cl_device_id*)alloca(sizeof(cl_device_id)*cnt_device_ids);
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, cnt_device_ids, device_ids, NULL) );
		for(int d=0; d<(int)cnt_device_ids; d++){
			if( dev_idx==cur_dev_idx )
				device_selected = device_ids[d];
			cur_dev_idx++;
		}
	}
	return device_selected;
}

void cl_helper_ValidateDeviceSelection(cl_device_id dev){
	if( dev==(cl_device_id)-1 ){
		fprintf(stderr, "Invalid device\n");
		exit(1);
	}
	size_t len;
	OCL_SAFE_CALL( clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &len) );
	char *cl_dev_name = (char*)alloca( len );
	OCL_SAFE_CALL( clGetDeviceInfo(dev, CL_DEVICE_NAME, len, cl_dev_name, NULL) );
	printf("Selected device: %s\n", cl_dev_name);
}

cl_program cl_helper_CreateBuildProgram(cl_context context, cl_device_id device, const char* src, const char *options){
	const char *all_sources[1] = {src};
	cl_int errno;
	flushed_printf("Creating program...");
	cl_program program = clCreateProgramWithSource(context, 1, all_sources, NULL, &errno);
	OCL_SAFE_CALL(errno);
	flushed_printf("Ok\n");

	flushed_printf("Building program...");
	errno = clBuildProgram(program, 1, &device, options, NULL, NULL);
	flushed_printf("Ok\n");

	if( errno!=CL_SUCCESS ){
		char log[10000];
		OCL_SAFE_CALL( clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL) );
		OCL_SAFE_CALL( clReleaseProgram(program) );
		puts(log);
		exit(EXIT_FAILURE);
	}
	return program;
}

/*
double loclGetEventExecTimeAndRelease(cl_event ev){
	cl_ulong ev_t_start, ev_t_finish;
	OCL_SAFE_CALL( clWaitForEvents(1, &ev) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_t_start, NULL) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_t_finish, NULL) );
	double time = (ev_t_finish-ev_t_start)/1000000000.0;
	OCL_SAFE_CALL( clReleaseEvent( ev ) );
	return time;
}
*/

int main(int argc, char* argv[]){
	printf("clmempatterns rel. 0.X\n");
	printf("developed by Elias Konstantinidis (ekondis@gmail.com)\n\n");

	if( argc<2 ){
		printf("parameters:\n");
		printf("clmempatterns {device index} [index magnitude [grid magnitude [workgroup magnitude [vector size]]]]\n");
		printf("All magnitudes are expressed as radix 2 logarithms of the respective quatities (e.g. magnitude of 10 implies 2^10=1024)\n\n");

		cl_helper_PrintAvailableDevices();

		exit(1);
	}
	
	cl_device_id selected_device_id = cl_helper_SelectDevice( atoi(argv[1]) );
	cl_helper_ValidateDeviceSelection( selected_device_id );

	unsigned int log2_indexes = argc<3 ? 22 : atoi(argv[2]);
	unsigned int log2_grid    = argc<4 ? 18 : atoi(argv[3]);
	unsigned int log2_wgroup  = argc<5 ?  8 : atoi(argv[4]);
	unsigned int vecsize      = argc<6 ?  2 : atoi(argv[5]); // 1, 2, 4, 8, 16

	printf("\nBenchmark parameters:\n");
	printf("index space %d\n", pow2(log2_indexes));
	printf("vector length %d\n", vecsize);
	printf("element space %d\n", pow2(log2_indexes)*vecsize);
	printf("Required memory %lu MB\n", pow2(log2_indexes)*vecsize*sizeof(int)/1024/1024);
	printf("grid space %d\n", pow2(log2_grid));
	printf("workgroup size %d\n", pow2(log2_wgroup));
	printf("total workgroups %d\n", pow2(log2_grid-log2_wgroup));
	printf("granularity %d\n", pow2(log2_indexes-log2_grid));
	
	printf("Full element space (bit format describing index space):\n");
	for(int i=log2_indexes-1; i>=0; i--)
		if( i % 10 == 0 )
			printf("%d", i / 10);
		else
			printf(" ");
	printf("\n");
	for(int i=log2_indexes-1; i>=0; i--)
		printf("%d", i % 10);
	printf("\n");
	for(int i=log2_indexes-1; i>=0; i--)
		printf("X");
	printf("\n");
	for(int i=log2_indexes-1; i>=0; i--)
		if( i<log2_grid )
			printf("P");
		else
			printf("I");
	printf("\n");
	for(int i=log2_indexes-1; i>=0; i--)
		if( i<log2_grid )
			if( i<log2_wgroup )
				printf("W");
			else
				printf("N");
		else
			printf("I");
	printf("\n\n");
	
	for(int stride_offset=0; stride_offset<(log2_indexes>log2_grid ? log2_grid+1 : 1); stride_offset++){
		printf("Stride offset %d:\n", stride_offset);
		for(int i=log2_indexes-1, p=log2_grid; i>=0; i--)
			if( i<log2_indexes-stride_offset && i>=log2_grid-stride_offset )
				printf("I");
			else {
				if( p<=log2_wgroup )
					printf("W");
				else
					printf("N");
				p--;
			}
		if( stride_offset == 0 ) printf(" Special case: All workitems access sequential elements in grid strides");
		if( stride_offset == log2_grid-log2_wgroup ) printf(" Special case: All workitems access sequential elements in workgroup strides");
		if( stride_offset == log2_grid ) printf(" Special case: A workitem accesses only sequential elements");
		printf("\n");
	}
puts(c_kernel);

	// Get platform ID
	cl_platform_id platform_id;
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform_id, NULL) );
	// Set context properties
	cl_context_properties ctxProps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0 };

	cl_int errno;
	// Create context
	cl_context context = clCreateContext(ctxProps, 1, &selected_device_id, ctxErrorCallback, NULL, &errno);
	OCL_SAFE_CALL(errno);

	// Get device limitations
	cl_ulong MAX_WORKGROUP_SIZE;
	OCL_SAFE_CALL( clGetDeviceInfo(selected_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &MAX_WORKGROUP_SIZE, NULL) );
//	printf("max wg size %ld\n", MAX_WORKGROUP_SIZE);

	// Create buffers
	flushed_printf("Creating buffer...");
	cl_mem dev_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pow2(log2_indexes)*vecsize*sizeof(int), NULL, &errno);
	OCL_SAFE_CALL(errno);
	flushed_printf("Ok\n");

	// Create command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, selected_device_id, CL_QUEUE_PROFILING_ENABLE, &errno);
	OCL_SAFE_CALL(errno);

	// Create and build program
	char c_compile_options[4096] = "-cl-std=CL1.1";
	//sprintf(compile_options, "-cl-fast-relaxed-math -cl-mad-enable -Doperator=%c -Dgranularity=%d -DSIMD_LEN=%d -Ddatatype=int%s -Dfdatatype=float%s %s -DSCALAR_TYPE=int -Dgrid_size=%zd -DTMP_S0=tmp%s -DWAVEFRONT_SIZE=%d -DVECTOR_REDUCTION(v)=(%s) ", op, granularity, simd_width, c_simd_len, c_simd_len, c_ddatatype, glWS[0], simd_width==1?"":".s0", (int)WAVEFRONT_SIZE, def_vecreduction);
	cl_program program = cl_helper_CreateBuildProgram(context, selected_device_id, c_kernel, c_compile_options);

	// Create kernels
	cl_kernel kernel_init = clCreateKernel(program, "initialize", &errno);
	OCL_SAFE_CALL(errno);
	cl_kernel kernel1 = clCreateKernel(program, "kernel1", &errno);
	OCL_SAFE_CALL(errno);

	// Initialize variables
	int index_space = pow2(log2_indexes);
	const size_t glWS[1] = {index_space/pow2(log2_indexes-log2_grid)};
	const size_t lcWS[1] = {pow2(log2_wgroup)};

	// Initialize buffer
	flushed_printf("Zeroing buffer...");
	OCL_SAFE_CALL( clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &dev_buffer) );
	OCL_SAFE_CALL( clSetKernelArg(kernel_init, 1, sizeof(cl_int), &index_space) );
//	errorCode = clEnqueueNDRangeKernel(cmd_queue, kernel_init, 1, NULL, glWS, lcWS, 0, NULL, NULL);
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel_init, 1, NULL, glWS, lcWS, 0, NULL, NULL) );
/*	switch( errorCode ){
		case CL_SUCCESS:
			time = loclGetEventExecTimeAndRelease(ev_wait);
			printf("Reduction kernel execution done (%f secs, %f GB/sec bandwidth)\n", time, 1.0*VEC_SIZ*sizeof(int)/(time*1000.0*1000.0*1000.0));
			break;
		case CL_OUT_OF_RESOURCES:
			fprintf(stderr, "Error: CL_OUT_OF_RESOURCES\n");
			exit(1);
		case CL_INVALID_WORK_GROUP_SIZE:
			fprintf(stderr, "Error: CL_INVALID_WORK_GROUP_SIZE\n");
			exit(1);
		default:
			OCL_SAFE_CALL( errorCode );
	}*/
	OCL_SAFE_CALL( clFinish(cmd_queue) );
	flushed_printf("Ok\n");
	
	// Release program and kernels
	OCL_SAFE_CALL( clReleaseKernel(kernel_init) );
	OCL_SAFE_CALL( clReleaseKernel(kernel1) );
	OCL_SAFE_CALL( clReleaseProgram(program) );

	// Release command queue
	OCL_SAFE_CALL( clReleaseCommandQueue(cmd_queue) );

	// Release buffer
	OCL_SAFE_CALL( clReleaseMemObject(dev_buffer) );

	// Release context
	OCL_SAFE_CALL( clReleaseContext(context) );
}

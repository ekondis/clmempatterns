#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <malloc.h>
#include <math.h>
#include <string.h>
#ifndef _MSC_VER
#include <alloca.h>
#endif
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

double sqr(double v){
	return v*v;
}

int compare_doubles(const void *a, const void *b){
	const double *v1 = (const double*)a;
	const double *v2 = (const double*)b;
	if( *v1 == *v2 ) return 0;
	return *v1 > *v2 ? 1 : -1;
}

void init_data(int *data, unsigned int len){
	for(int i=0; i<(int)len; i++)
		data[i] = 0;
}

void flushed_printf(const char* format, ...){
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

void show_progress_init(int length){
	flushed_printf("[");
	for(int i=0; i<length; i++)
		flushed_printf(" ");
	flushed_printf("]");
	for(int i=0; i<=length; i++)
		flushed_printf("\b");
}

void show_progress_step(int domove, char newchar){
	flushed_printf("%c", newchar);
	if( !domove )
		flushed_printf("\b");
}

void show_progress_done(void){
	flushed_printf("\n");
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
					dev_type_str = "Other";
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
	
	OCL_SAFE_CALL( clGetDeviceInfo(dev, CL_DRIVER_VERSION, 0, NULL, &len) );
	char *cl_driver_version = (char*)alloca( len );
	OCL_SAFE_CALL( clGetDeviceInfo(dev, CL_DRIVER_VERSION, len, cl_driver_version, NULL) );
	
	cl_platform_id platform;
	OCL_SAFE_CALL( clGetDeviceInfo(dev, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL) );

	OCL_SAFE_CALL( clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &len) );
	char *cl_plf_name = (char*)alloca( len );
	OCL_SAFE_CALL( clGetPlatformInfo(platform, CL_PLATFORM_NAME, len, cl_plf_name, NULL) );

	printf("Selected platform: %s\n", cl_plf_name);
	printf("Selected device  : %s\n", cl_dev_name);
	printf("Driver version   : %s\n", cl_driver_version);
}

cl_program cl_helper_CreateBuildProgram(cl_context context, cl_device_id device, const char* src, const char *options){
	const char *all_sources[1] = {src};
	cl_int errno;
//	show_progress_step(0, '\\');
//	flushed_printf("Creating program... ");
	cl_program program = clCreateProgramWithSource(context, 1, all_sources, NULL, &errno);
	OCL_SAFE_CALL(errno);
//	flushed_printf("Ok\n");
	show_progress_step(0, '>');

//	flushed_printf("Building program... ");
	errno = clBuildProgram(program, 1, &device, options, NULL, NULL);
//	flushed_printf("Ok\n");
	show_progress_step(1, '#');

	if( errno!=CL_SUCCESS ){
		char log[10000];
		OCL_SAFE_CALL( clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL) );
		OCL_SAFE_CALL( clReleaseProgram(program) );
		fprintf(stderr, "%s", log);
		exit(EXIT_FAILURE);
	}
	return program;
}

double cl_helper_GetExecTimeAndRelease(cl_event ev){
	cl_ulong ev_t_start, ev_t_finish;
	OCL_SAFE_CALL( clWaitForEvents(1, &ev) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_t_start, NULL) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_t_finish, NULL) );
	double time = (ev_t_finish-ev_t_start)/1000000000.0;
	OCL_SAFE_CALL( clReleaseEvent( ev ) );
	return time;
}

int main(int argc, char* argv[]){
	printf("clmempatterns rel. 0.2git\n");
	printf("Developed by Elias Konstantinidis (ekondis@gmail.com)\n\n");

	// Parameters
	cl_device_id selected_device_id = (cl_device_id)-1;
	unsigned int log2_indexes = 22;
	unsigned int log2_grid    = 18;
	unsigned int log2_wgroup  =  8;
	unsigned int vecsize      =  2; // 1, 2, 4, 8, 16
	char *foutput = NULL;
	int b_use_host_buffer     = 0; // default: device allocated buffer

	// parse arguments
	int arg_count = 0;
	for(int i=1; i<argc; i++) {
		if( (strcmp(argv[i], "-h")==0) || (strcmp(argv[i], "--help")==0) ) {
			selected_device_id = (cl_device_id)-1;
			break;
		} else if( (strcmp(argv[i], "-H")==0) || (strcmp(argv[i], "--host")==0) ) {
			b_use_host_buffer = 1;
		} else if( (strcmp(argv[i], "-o")==0) || (strcmp(argv[i], "--output")==0) ) {
			if( ++i>=argc ){
				selected_device_id = (cl_device_id)-1;
				break;
			}
			foutput = (char*)alloca(sizeof(char)*strlen(argv[i]));
			strcpy(foutput, argv[i]);
		} else {
			unsigned long value = strtoul(argv[i], NULL, 10);
			switch( arg_count ){
				// device selection
				case 0:
					selected_device_id = cl_helper_SelectDevice( value );
					arg_count++;
					break;
				// index magnitude
				case 1:
					log2_indexes = strtoul(argv[2], NULL, 10);
					arg_count++;
					break;
				// grid magnitude
				case 2:
					log2_grid    = strtoul(argv[3], NULL, 10);
					arg_count++;
					break;
				// workgroup size magnitude
				case 3:
					log2_wgroup  = strtoul(argv[4], NULL, 10);
					arg_count++;
					break;
				// vector size
				case 4:
					vecsize      = strtoul(argv[5], NULL, 10); // 1, 2, 4, 8, 16
					arg_count++;
					break;
				default:
					selected_device_id = (cl_device_id)-1;
			}
		}
	}

	if( selected_device_id == (cl_device_id)-1 ){
		printf("Usage: clmempatterns [options] {device index} [index magnitude [grid magnitude [workgroup magnitude [vector size]]]]\n");
		printf("* All magnitudes are expressed in base-2 logarithmic scales (e.g. 10 implies 2^10=1024)\n\n");
		printf("Options:\n"
			"-h or --help          Show this message\n"
			"-H or --host          Use host allocated buffer (CL_MEM_ALLOC_HOST_PTR)\n"
			"-o or --output <file> Save CSV output to <file>\n\n");

		cl_helper_PrintAvailableDevices();

		exit(1);
	}

	cl_helper_ValidateDeviceSelection( selected_device_id );

	const unsigned int max_log2_stride = log2_indexes>log2_grid ? log2_grid : 0;
	
	if( log2_indexes<log2_grid ){
		fprintf(stderr, "\nERROR: Grid magnitude cannot exceed index magnitude (%d>%d).\n", log2_grid, log2_indexes);
		exit(1);
	}

	printf("\nBenchmark parameters:\n");
	printf("index space     : %d\n", pow2(log2_indexes));
	printf("vector length   : %d (type: int%c)\n", vecsize, vecsize==1 ? ' ' : '0'+vecsize);
	//printf("element space   : %d\n", pow2(log2_indexes)*vecsize);
	{
		unsigned long int req_mem = pow2(log2_indexes)*vecsize*sizeof(int)/1024;
		char req_mem_unit = 'K';
		if( req_mem>=1024*8 ){
			req_mem /= 1024;
			req_mem_unit = 'M';
		}
		if( req_mem>=1024*8 ){
			req_mem /= 1024;
			req_mem_unit = 'G';
		}
		printf("required memory : %lu %cB\n", req_mem, req_mem_unit);
	}
	printf("grid space      : %d (%d workgroups)\n", pow2(log2_grid), pow2(log2_grid-log2_wgroup));
	printf("workgroup size  : %d\n", pow2(log2_wgroup));
	printf("total workgroups: %d\n", pow2(log2_grid-log2_wgroup));
	printf("granularity     : %d\n", pow2(log2_indexes-log2_grid));
	printf("allocated buffer: %s\n", b_use_host_buffer ? "Host" : "Device");

//puts(c_kernel);

	printf("\nPlatform initialization:\n");
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

	// Create buffers
	flushed_printf("Creating buffer... ");
	cl_mem_flags buf_flags = CL_MEM_READ_WRITE;
	if( b_use_host_buffer )
		buf_flags |= CL_MEM_ALLOC_HOST_PTR;
	cl_mem dev_buffer = clCreateBuffer(context, buf_flags, pow2(log2_indexes)*vecsize*sizeof(cl_int), NULL, &errno);
	OCL_SAFE_CALL(errno);
	flushed_printf("Ok\n");
	
	// Create command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, selected_device_id, CL_QUEUE_PROFILING_ENABLE, &errno);
	OCL_SAFE_CALL(errno);

	// Initialize buffer
	/*cl_int *temp = (cl_int*)clEnqueueMapBuffer(cmd_queue, dev_buffer, CL_TRUE, CL_MAP_WRITE, 0, pow2(log2_indexes)*vecsize*sizeof(cl_int), 0, NULL, NULL, &errno);
	OCL_SAFE_CALL(errno);
	for(int i=0; i<pow2(log2_indexes)*vecsize; i++)
		temp[i] = 0;
	clEnqueueUnmapMemObject(cmd_queue, dev_buffer, temp, 0, NULL, NULL);*/

	// Create and build program
	flushed_printf("Building programs... ");
	char c_compile_options[4096];// = "-cl-std=CL1.1 -DDATATYPE=int2 -DSTRIDE_ORDER=18 -DGRANULARITY_ORDER=4";
	cl_program *programs = (cl_program*)alloca(sizeof(cl_program)*(max_log2_stride+1));
	show_progress_init(max_log2_stride+1);
	for(int stride_offset=0; stride_offset<=(int)max_log2_stride; stride_offset++){
		sprintf(c_compile_options, "-cl-std=CL1.1 -DDATATYPE=int%c -DSTRIDE_ORDER=%d -DGRANULARITY_ORDER=%d", (vecsize==1 ? ' ' : ('0'+vecsize)), stride_offset, log2_indexes-log2_grid);
		//printf("Using options %s\n", c_compile_options);
		programs[stride_offset] = cl_helper_CreateBuildProgram(context, selected_device_id, c_kernel, c_compile_options);
	}
	show_progress_done();
	//flushed_printf("Ok\n");

	// Create kernels
	flushed_printf("Creating kernels... ");
	cl_kernel *kernels_init = (cl_kernel*)alloca(sizeof(cl_kernel)*(max_log2_stride+1));
	cl_kernel *kernels1 = (cl_kernel*)alloca(sizeof(cl_kernel)*(max_log2_stride+1));
	for(int stride_offset=0; stride_offset<=(int)max_log2_stride; stride_offset++){
		kernels_init[stride_offset] = clCreateKernel(programs[stride_offset], "initialize", &errno);
		OCL_SAFE_CALL(errno);
		kernels1[stride_offset] = clCreateKernel(programs[stride_offset], "kernel1", &errno);
		OCL_SAFE_CALL(errno);
	}
	flushed_printf("Ok\n");

	// Initialize variables
	cl_int index_space = pow2(log2_indexes);
	const cl_int zero=0;//, one=1;
	const size_t glWS[1] = {index_space/pow2(log2_indexes-log2_grid)};
	const size_t lcWS[1] = {pow2(log2_wgroup)};
	cl_event ev_wait;

	// Initialize buffer
	flushed_printf("Zeroing buffer... ");
	OCL_SAFE_CALL( clSetKernelArg(kernels_init[0], 0, sizeof(cl_mem), &dev_buffer) );
	OCL_SAFE_CALL( clSetKernelArg(kernels_init[0], 1, sizeof(cl_int), &index_space) );
	OCL_SAFE_CALL( clSetKernelArg(kernels_init[0], 2, sizeof(cl_int), &zero) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernels_init[0], 1, NULL, glWS, lcWS, 0, NULL, NULL) );
	OCL_SAFE_CALL( clFinish(cmd_queue) );
	flushed_printf("Ok\n");

	printf("\nExperimental execution:\n");
	double *total_times = (double*)alloca(sizeof(double)*(max_log2_stride+1));
	// iterate over stride values
	const int REPETITIONS = 16;
	flushed_printf("Running... ");
	show_progress_init( max_log2_stride+1 );
//	double max_variation_coeff = 0.0;
	for(int stride_offset=0; stride_offset<=(int)max_log2_stride; stride_offset++){
		show_progress_step(0, '>');
		//printf("Stride offset %2d:", stride_offset);
		//printf("\\|/-\n");

		// warm up
		//flushed_printf("Warmimg up... ");
		OCL_SAFE_CALL( clSetKernelArg(kernels1[stride_offset], 0, sizeof(cl_mem), &dev_buffer) );
		OCL_SAFE_CALL( clSetKernelArg(kernels1[stride_offset], 1, sizeof(cl_int), &index_space) );
		OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernels1[stride_offset], 1, NULL, glWS, lcWS, 0, NULL, NULL) );
		//cl_helper_GetExecTimeAndRelease(ev_wait);
		OCL_SAFE_CALL( clFinish(cmd_queue) );
		//flushed_printf("Ok ");
	//	printf("Done in %f msecs (%.3f GB/sec bandwidth)\n", 1000.0*time, pow2(log2_indexes)*vecsize*sizeof(int)/(time*1000.0*1000.0*1000.0));

		// run benchmarks multiple times
		double *times = (double*)alloca(sizeof(double)*REPETITIONS);
		OCL_SAFE_CALL( clSetKernelArg(kernels1[stride_offset], 0, sizeof(cl_mem), &dev_buffer) );
		OCL_SAFE_CALL( clSetKernelArg(kernels1[stride_offset], 1, sizeof(cl_int), &index_space) );
		int do_run_experiment = 1;
		while( do_run_experiment ){
			for(int i=0; i<REPETITIONS; i++){
				const char chr_progress[] = "\\|/-";
				show_progress_step(0, chr_progress[i%4]);
				OCL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernels1[stride_offset], 1, NULL, glWS, lcWS, 0, NULL, &ev_wait) );
				times[i] = cl_helper_GetExecTimeAndRelease(ev_wait);
			}
			qsort(times, REPETITIONS, sizeof(times[0]), compare_doubles);
			const double median_time = REPETITIONS % 2 ? times[REPETITIONS/2] : (times[REPETITIONS/2-1]+times[REPETITIONS/2])/2;
			double average_time = 0., variance = 0.;
			for(int i=0; i<REPETITIONS; i++)
				average_time += times[i];
			average_time /= REPETITIONS;
			for(int i=0; i<REPETITIONS; i++)
				variance += sqr(times[i]-average_time);
			variance /= REPETITIONS;
			double variation_coeff = sqrt(variance)/average_time;
			const double VAR_COEFF_THRESHOLD = 0.25;
			if( variation_coeff<VAR_COEFF_THRESHOLD )
				do_run_experiment = 0;
			//else flushed_printf("%e\n", variation_coeff);//show_progress_step(1, 'E');
			total_times[stride_offset] = median_time;//average_time;
		}
		/*if( max_variation_coeff<variation_coeff )
			max_variation_coeff = variation_coeff;*/
		show_progress_step(1, '#');
	}
	show_progress_done();
	
	//flushed_printf("Ok\n");
	/*if(max_variation_coeff>VAR_COEFF_THRESHOLD){
		fprintf(stderr, "\nERROR: Variation coefficient of execution time (%5.3f) exceeded threshold (%5.3f).\n", max_variation_coeff, VAR_COEFF_THRESHOLD);
		fprintf(stderr, "{");
		for(int i=0; i<REPETITIONS; i++)
			fprintf(stderr, "%e, ", times[i]);
		fprintf(stderr, "}\n");
		fprintf(stderr, "{");
		for(int i=0; i<REPETITIONS; i++)
			fprintf(stderr, "%6.2f%%, ", 100.*(times[i]-average_time)/average_time);
		fprintf(stderr, "}\n");
		fprintf(stderr, "{");
		for(int i=0; i<REPETITIONS; i++)
			fprintf(stderr, "%6.2f%%, ", 100.*(times[i]-median_time)/median_time);
		fprintf(stderr, "}\n");
		fprintf(stderr, "ERROR: Average (%e) Median (%e)\n", average_time, median_time);	
		exit(1);
	}*/

	// Print results on screen
	printf("\nSummary:");
	for(int stride_offset=0; stride_offset<=(int)max_log2_stride; stride_offset++){
		printf("\nStride magnitude %2d: Bandwidth %7.3f GB/sec (avg time %10f msecs)", 
			stride_offset, 
			pow2(log2_indexes)*vecsize*sizeof(int)/(total_times[stride_offset]*1000.0*1000.0*1000.0), 
			1000.0*total_times[stride_offset]);
		if( stride_offset == log2_grid ) printf(" *Special case: All workitems access sequential elements doing grid strides");
		if( stride_offset == log2_wgroup ) printf(" *Special case: All workitems access sequential elements doing workgroup strides");
		if( stride_offset == 0 ) printf(" *Special case: A workitem accesses elements only sequentially");
	}
	printf("\n");
	
	// Save output if requested
	if( foutput ){
		printf("Writing results to %s\n", foutput);
		FILE *of = fopen(foutput, "w");
		if (of == NULL) {
			fprintf(stderr, "Can't open output file %s!\n", foutput);
			exit(1);
		}
		/*fprintf(of, "stride,    ex_time, bandwidth\n");
		for(int stride_offset=0; stride_offset<=(int)max_log2_stride; stride_offset++)
			fprintf(of, 
				"%6d, %10f, %9.3f\n", 
				stride_offset, 
				1000.0*total_times[stride_offset], 
				pow2(log2_indexes)*vecsize*sizeof(int)/(total_times[stride_offset]*1000.0*1000.0*1000.0));*/
		for(int stride_offset=0; stride_offset<(int)max_log2_stride; stride_offset++)
			fprintf(of, "%7d, ", stride_offset);
		fprintf(of, "%7d ", (int)max_log2_stride);
		fprintf(of, "\n");
		for(int stride_offset=0; stride_offset<(int)max_log2_stride; stride_offset++)
			fprintf(of, "%7.3f, ", pow2(log2_indexes)*vecsize*sizeof(int)/(total_times[stride_offset]*1000.0*1000.0*1000.0));
		fprintf(of, "%7.3f ", pow2(log2_indexes)*vecsize*sizeof(int)/(total_times[max_log2_stride]*1000.0*1000.0*1000.0));
		fprintf(of, "\n");
		fclose(of);
	}
	
	// Release program and kernels
	for(int stride_offset=0; stride_offset<=(int)max_log2_stride; stride_offset++){
		OCL_SAFE_CALL( clReleaseKernel(kernels_init[stride_offset]) );
		OCL_SAFE_CALL( clReleaseKernel(kernels1[stride_offset]) );
		OCL_SAFE_CALL( clReleaseProgram(programs[stride_offset]) );
	}

	// Release command queue
	OCL_SAFE_CALL( clReleaseCommandQueue(cmd_queue) );

	// Release buffer
	OCL_SAFE_CALL( clReleaseMemObject(dev_buffer) );

	// Release context
	OCL_SAFE_CALL( clReleaseContext(context) );
}

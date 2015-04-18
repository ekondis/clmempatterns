#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
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

	unsigned int log2_indexes = argc<3 ? 24 : atoi(argv[2]);
	unsigned int log2_grid    = argc<4 ? 18 : atoi(argv[3]);
	unsigned int log2_wgroup  = argc<5 ?  8 : atoi(argv[4]);
	unsigned int vecsize      = argc<6 ?  2 : atoi(argv[5]); // 1, 2, 4, 8, 16

	printf("\nindex space %d\n", pow2(log2_indexes));
//	printf("vector memory size %d\n", pow2(log2_vecsize)*sizeof(int));
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
}

#include <stdio.h>
#include <stdlib.h>

unsigned int pow2(unsigned int v){
	return 1 << v;
}

int main(int argc, char* argv[]){
	printf("clmempatterns rel. 0.X\n");
	printf("developed by Elias Konstantinidis\n\n");

	printf("parameters:\n");
	printf("clmempatterns {device index} [index magnitude [grid magnitude [workgroup magnitude [vector size]]]]\n");
	printf("All magnitudes are expressed as radix 2 logarithm integers of the respective quatities.\n");
	
	unsigned int log2_indexes = argc<3 ? 24 : atoi(argv[2]);
	unsigned int log2_grid    = argc<4 ? 18 : atoi(argv[3]);
	unsigned int log2_wgroup  = argc<5 ?  8 : atoi(argv[4]);
	unsigned int vecsize      = argc<6 ?  2 : atoi(argv[5]); // 1, 2, 4, 8, 16

	printf("index space %d\n", pow2(log2_indexes));
//	printf("vector memory size %d\n", pow2(log2_vecsize)*sizeof(int));
	printf("element space %d\n", pow2(log2_indexes)*vecsize);
	printf("Required memory %d MB\n", pow2(log2_indexes)*vecsize*sizeof(int)/1024/1024);
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

}


// required definitions
//#define DATATYPE int2
//#define STRIDE_ORDER 18
//#define GRANULARITY_ORDER 4
//#define OPT_ZEROCOPY not needed
//#define OPT_NOTDUMMY

#define CMD_HELPER(FUNC, NAME) FUNC ## _ ## NAME
#define CMD(FUNC, NAME) CMD_HELPER(FUNC, NAME)

#define STRIDE (1 << STRIDE_ORDER)
#define GRANULARITY (1 << GRANULARITY_ORDER) 

int reduce_int(int v)    { return v; }
int reduce_int2(int2 v)  { return v.x+v.y; }
int reduce_int4(int4 v)  { return v.x+v.y+v.z+v.w; }
int reduce_int8(int8 v)  { return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7; }
int reduce_int16(int16 v){ return v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7+v.s8+v.s9+v.sA+v.sB+v.sC+v.sD+v.sE+v.sF; }

__kernel void initialize(__global DATATYPE *data, const uint n, const int v) {
	const uint id = get_global_id(0);
	const uint stride = get_global_size(0);
	uint offset = 0;
	while( id+offset<n ){
		data[id+offset] = (DATATYPE)v;
		offset += stride;
	}
}

__kernel void kernel1(__global DATATYPE *data, const uint n) {
	// Get our global thread ID
	const uint id = get_global_id(0);
	const uint  low_order_id = id & ( (uint)(STRIDE - 1));
	const uint high_order_id = id & (~(uint)(STRIDE - 1));
	const uint index = (high_order_id << GRANULARITY_ORDER) | low_order_id;
	const int localid = get_local_id(0);
	const int group_size = get_local_size(0);
/*	__local uint lcount;
	barrier(CLK_LOCAL_MEM_FENCE);
	if( localid==0 )
		lcount = 0;*/
	// Make sure we do not go out of bounds
//	DATATYPE tmp = (DATATYPE)0;
//	#pragma unroll
	for(int i=0; i<GRANULARITY; i++){
		DATATYPE v = data[index+i*STRIDE];
		int v_sum = CMD(reduce, DATATYPE)(v);
		if( v_sum>0 )
			data[index+i*STRIDE] = (DATATYPE)0;
		//tmp = tmp+data[index+i*STRIDE];
		//data[index+i*STRIDE] = (DATATYPE)(index+i*STRIDE);
	}
/*	if( count )
		atomic_add(&lcount, count);
	barrier(CLK_LOCAL_MEM_FENCE);
	// Atomic reduce in global memory
	if(localid==0 && lcount){
		atomic_add((__global int*)result, lcount);
	}*/
}

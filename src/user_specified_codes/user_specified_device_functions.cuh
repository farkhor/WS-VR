#ifndef	USER_SPECIFIED_DEVICE_FUNCTIONS_CUH
#define	USER_SPECIFIED_DEVICE_FUNCTIONS_CUH

#include "user_specified_global_configurations.h"
#include "user_specified_structures.h"



/**************************************
 *  PROCESSING FUNCTIONS
 **************************************/

// At every iteration, you need to initialize shared memory with vertices.
// This function is executed for each and every vertex.
inline __host__ __device__ void init_compute(
		volatile Vertex* local_V,	// Address of the corresponding vertex in shared memory.
		Vertex* V	) {	// Address of the previous version of the vertex.

#ifdef BFS
	local_V->distance = V->distance;
#endif

#ifdef SSSP
	local_V->distance = V->distance;
#endif

#ifdef PR
	local_V->rank = 0;
#endif

}


// In below, each thread computes a result based on edge data and writes to its own specific shared memory.
// This function is executed for each and every edge.
inline __host__ __device__ void compute_local(
		Vertex SrcV,	// Source vertex in global memory.
		const Vertex_static* SrcV_static,	// Source Vertex_static in global memory. Dereferencing this pointer if it's not defined causes error.
		const Edge* E,	// Edge in global memory. Dereferencing this pointer if it's not defined cause error.
		volatile Vertex* thread_V_in_shared,	// Thread's specific shared memory region to store the result of the local computation.
		Vertex* refV	) {	// Value of the corresponding (destination) vertex inside shared memory.

#ifdef BFS
	thread_V_in_shared->distance = SrcV.distance + 1;
#endif

#ifdef SSSP
	thread_V_in_shared->distance = SrcV.distance + E->weight;
#endif

#ifdef PR
	unsigned int nbrsNum = SrcV_static->NbrsNum;
	thread_V_in_shared->rank = ( nbrsNum != 0 ) ? ( SrcV.rank / nbrsNum ) : 0;
#endif

}

// Reduction function that is performed for every pair of neighbors of a vertex.
inline __host__ __device__ void compute_reduce(
		volatile Vertex* thread_V_in_shared,
		Vertex* next_thread_V_in_shared	) {

#ifdef BFS
	if ( thread_V_in_shared->distance > next_thread_V_in_shared->distance )
		thread_V_in_shared->distance = next_thread_V_in_shared->distance;
#endif

#ifdef SSSP
	if ( thread_V_in_shared->distance > next_thread_V_in_shared->distance)
		thread_V_in_shared->distance = next_thread_V_in_shared->distance;
#endif

#ifdef PR
	thread_V_in_shared->rank += next_thread_V_in_shared->rank;
#endif

}


// Below function signals the caller (and consequently the host) if the vertex content should be replaced with the newly calculated value.
// This function is performed by one virtual lane in the virtual warp.
inline __host__ __device__ bool update_condition (	volatile Vertex* computed_V,
		volatile Vertex* previous_V	) {

#ifdef BFS
	return ( computed_V->distance < previous_V->distance );
#endif

#ifdef SSSP
	return ( computed_V->distance < previous_V->distance );
#endif

#ifdef PR
	computed_V->rank = (1-PR_DAMPING_FACTOR) + computed_V->rank * PR_DAMPING_FACTOR;	// Or you can replace this expression by fused multiply-add.
	return ( fabs( computed_V->rank - previous_V->rank) > PR_TOLERANCE );
#endif

}



#endif	// USER_SPECIFIED_DEVICE_FUNCTIONS_CUH

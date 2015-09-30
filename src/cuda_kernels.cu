#include <cmath>

#include "cuda_kernels.cuh"
#include "utils/CUDAErrorCheck.cuh"
#include "user_specified_codes/user_specified_global_configurations.h"
#include "user_specified_codes/user_specified_device_functions.cuh"


/**************************************************************************************************
 * INBOX UNLOADING KERNEL.
 **************************************************************************************************/

template < typename idxT, typename vT >
__global__ void dev_unload_inbox( vT* VertexValue, const uint loadSize, const idxT* inboxIndices, const vT* inboxVertices ) {

	const uint globalTID = threadIdx.x + blockIdx.x * blockDim.x;
	if( globalTID < loadSize )
		VertexValue[ inboxIndices[ globalTID ] ] = inboxVertices[ globalTID ];

}

void cuda_kernels::unload_inbox( Vertex* VertexValue, const uint loadSize, const uint* inboxIndices, const Vertex* inboxVertices, cudaStream_t sss, const uint offset ) {

	const uint gridDim = std::ceil( static_cast<float>( loadSize ) / COMPILE_TIME_DETERMINED_BLOCK_SIZE );
	if( gridDim != 0 )
		dev_unload_inbox<<< gridDim, COMPILE_TIME_DETERMINED_BLOCK_SIZE, 0, sss >>>( VertexValue, loadSize, inboxIndices + offset, inboxVertices + offset );
	CUDAErrorCheck( cudaPeekAtLastError() );

}


/**************************************************************************************************
 * FILLING OUTBOX KERNEL. USED ONLY IN MS.
 **************************************************************************************************/

template < typename idxT, typename vT >
__global__ void dev_fill_outbox( const vT* VertexValue, const uint loadSize, const idxT* outboxIndices, vT* outboxVertices ) {

	const uint globalTID = threadIdx.x + blockIdx.x * blockDim.x;
	if( globalTID < loadSize )
		outboxVertices[ globalTID ] = VertexValue[ outboxIndices[ globalTID ] ];

}

void cuda_kernels::fill_outbox( const Vertex* VertexValue, const uint loadSize, const uint* outboxIndices, Vertex* outboxVertices, cudaStream_t sss ) {

	const uint gridDim = std::ceil( static_cast<float>( loadSize ) / COMPILE_TIME_DETERMINED_BLOCK_SIZE );
	if( gridDim != 0 )
		dev_fill_outbox<<< gridDim, COMPILE_TIME_DETERMINED_BLOCK_SIZE, 0, sss >>>( VertexValue, loadSize, outboxIndices, outboxVertices );
	CUDAErrorCheck( cudaPeekAtLastError() );

}


/**************************************************************************************************
 * DISTRIBUTE OUTBOX KERNEL. USED ONLY IN VR/VRONLINE.
 **************************************************************************************************/

template < typename idxT, typename vT >
__global__ void dev_distribute_outbox( const uint* loadSizePtr, const idxT* srcIndices, const vT* srcVertices, idxT* dstIndices, vT* dstVertices ) {

	const uint globalTID = threadIdx.x + blockIdx.x * blockDim.x;
	if( globalTID < (*loadSizePtr) ) {
		dstIndices[ globalTID ] = srcIndices[ globalTID ];
		dstVertices[ globalTID ] = srcVertices[ globalTID ];
	}

}

void cuda_kernels::distribute_outbox( const uint maxLoadSize, const uint* loadSizePtr, const uint* srcIndices, const Vertex* srcVertices, uint* dstIndices, Vertex* dstVertices, cudaStream_t sss, uint dstOffset ) {

	const uint gridDim = std::ceil( static_cast<float>( maxLoadSize ) / COMPILE_TIME_DETERMINED_BLOCK_SIZE );
	if( gridDim != 0 )
		dev_distribute_outbox<<< gridDim, COMPILE_TIME_DETERMINED_BLOCK_SIZE, 0, sss >>>( loadSizePtr, srcIndices, srcVertices, dstIndices + dstOffset, dstVertices + dstOffset );
	CUDAErrorCheck( cudaPeekAtLastError() );

}


/**************************************************************************************************
 * WARP SEGMENTATION DEVICE AND GLOBAL FUNCTIONS.
 **************************************************************************************************/

// Device function to find the belonging vertex index for the thread's assigned edge via a binary search.
template< typename idxT >
__device__ __inline__ uint find_belonging_vertex_index_inside_warp( volatile idxT* edgesIndicesShared, const idxT currentEdgeIdx ) {

	idxT startIdx = 0;
	idxT endIdx = WARP_SIZE;
	#pragma unroll 5
	for( int iii = 0; iii < WARP_SIZE_SHIFT; ++iii ) {
		idxT middle = ( startIdx + endIdx ) >> 1;
		if( currentEdgeIdx < edgesIndicesShared[ middle ] )
			endIdx = middle;
		else
			startIdx = middle;
	}
	return startIdx;

}


// The main graph processing CUDA kernel for multiple devices.
template< inter_device_comm InterDeviceCommunication >	// The mode of inter device communication.
__global__ void dev_Warp_Segmentation_multi_gpu(
		const uint nVerticesToProcess,
		const uint* vertexIndices,
		const uint* edgesIndices,
		Vertex* VertexValue,
		const Edge* EdgeValue,
		const Vertex_static* VertexValueStatic,
		int* devFinished,
		const uint vertexOffset = 0,
		const uint edgeOffset = 0,
		uint* movingIndex = NULL,
		uint* outboxIndices = NULL,
		Vertex* outboxVertices = NULL
		) {

	__shared__ Vertex fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	__shared__ Vertex locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint fetchedEdgesIndices[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint fetchedEdgesIndicesInitial[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	const uint tidWithinCTA = threadIdx.x;
	uint laneID; asm( "mov.u32 %0, %laneid;" : "=r"(laneID) );	//const uint laneID = tidWithinCTA & ( WARP_SIZE - 1 );
	const uint inDeviceVertexID = tidWithinCTA + blockIdx.x * blockDim.x;
	const uint multiDeviceVertexID = inDeviceVertexID + vertexOffset;
	if( inDeviceVertexID >= nVerticesToProcess )
		return;

	// Initialize vertices.
	init_compute( fetchedVertexValues + tidWithinCTA, VertexValue + multiDeviceVertexID );
	const uint warpInDeviceVertexOffset = inDeviceVertexID & ( ~ ( WARP_SIZE - 1 ) );

	uint endEdgeIdx;
	if( InterDeviceCommunication == VR ) {
		fetchedEdgesIndicesInitial[ tidWithinCTA ] = edgesIndices[ inDeviceVertexID ] - edgeOffset;
		fetchedEdgesIndices[ tidWithinCTA ] = fetchedEdgesIndicesInitial[ tidWithinCTA ] & 0x7FFFFFFF;
		endEdgeIdx = ( edgesIndices[ warpInDeviceVertexOffset + WARP_SIZE ] & 0x7FFFFFFF ) - edgeOffset;
	}
	else {
		fetchedEdgesIndices[ tidWithinCTA ] = edgesIndices[ inDeviceVertexID ] - edgeOffset;
		endEdgeIdx = edgesIndices[ warpInDeviceVertexOffset + WARP_SIZE ] - edgeOffset;
	}

	const uint warpOffsetWithinCTA = tidWithinCTA & ( ~ ( WARP_SIZE - 1 ) );
	const uint startEdgeIdx = fetchedEdgesIndices[ warpOffsetWithinCTA ];	// first place in the warp specific shared memory.

	// Compute and reduce using source vertices.
	for( uint currentEdgeIdx = startEdgeIdx + laneID;
			currentEdgeIdx < endEdgeIdx;
			currentEdgeIdx += WARP_SIZE ) {

		const uint targetVertexIndex = vertexIndices[ currentEdgeIdx ];
		const Vertex srcValue = VertexValue[ targetVertexIndex ];

		const uint belongingVertexIdx = find_belonging_vertex_index_inside_warp(
				fetchedEdgesIndices + warpOffsetWithinCTA,
				currentEdgeIdx );

		// reduction parameters.
		const uint nZerosLowerThanPositionInSegmentInclusive = min( laneID, currentEdgeIdx - fetchedEdgesIndices[ warpOffsetWithinCTA + belongingVertexIdx ] );
		const uint nZerosHigherThanPositionInSegment = min( ( (belongingVertexIdx==31)?endEdgeIdx:fetchedEdgesIndices[ warpOffsetWithinCTA + belongingVertexIdx + 1 ] ) - currentEdgeIdx, WARP_SIZE - laneID );
		const uint segmentSize = nZerosHigherThanPositionInSegment + nZerosLowerThanPositionInSegmentInclusive;
		uint iii = 	( segmentSize > 16 ) ? 32 :
					( segmentSize >  8 ) ? 16 :
					( segmentSize >  4 ) ?  8 :
					( segmentSize >  2 ) ?  4 :
					segmentSize;

		// compute local
		compute_local(
				srcValue,
				VertexValueStatic + targetVertexIndex,
				EdgeValue + currentEdgeIdx,
				locallyComputedVertexValues + tidWithinCTA,
				fetchedVertexValues + warpOffsetWithinCTA + belongingVertexIdx );

		// first reduction iteration.
		if( iii != 1 ) {
			const uint jjj = iii >> 1;
			if( ( nZerosLowerThanPositionInSegmentInclusive + jjj ) < segmentSize )
				compute_reduce( locallyComputedVertexValues + tidWithinCTA, locallyComputedVertexValues + tidWithinCTA + jjj );
			iii = jjj;
		}
				// rest of iterations.
		while( iii != 1 ) {
			const uint jjj = iii >> 1;
			if( nZerosLowerThanPositionInSegmentInclusive < jjj )
				compute_reduce( locallyComputedVertexValues + tidWithinCTA, locallyComputedVertexValues + tidWithinCTA + jjj );
			iii = jjj;
		}

		// final reduction by the first lane in the segment.
		if( nZerosLowerThanPositionInSegmentInclusive == 0 ) {
			compute_reduce( fetchedVertexValues + warpOffsetWithinCTA + belongingVertexIdx, locallyComputedVertexValues + tidWithinCTA );
		}

	}

	// Update vertices.
	if( InterDeviceCommunication == MS || InterDeviceCommunication == ALL ) {

		if( update_condition( fetchedVertexValues + tidWithinCTA, VertexValue + multiDeviceVertexID  ) ) {
			(*devFinished) = 1;
			VertexValue[ multiDeviceVertexID ] = fetchedVertexValues[ tidWithinCTA ];
		}

	}
	if( InterDeviceCommunication == VRONLINE ) {

		const bool updated = update_condition( fetchedVertexValues + tidWithinCTA, VertexValue + multiDeviceVertexID  );

		if( __any( updated ) ) {

			const uint warpBallot = __ballot( updated );
			const int nUpdates = __popc( warpBallot );
			int indexPosition;
			if( laneID == 0 ) {
				(*devFinished) = 1;
				indexPosition = atomicAdd( movingIndex, nUpdates );
			}
			indexPosition = __shfl( indexPosition, 0 );
			uint laneMask; asm( "mov.u32 %0, %lanemask_lt;" : "=r"(laneMask) );	//const uint laneMask =  0xFFFFFFFF >> ( WARP_SIZE - laneID ) ;
			const int positionToWrite = indexPosition + __popc( warpBallot & laneMask );
			if( updated ) {
				VertexValue[ multiDeviceVertexID ] = fetchedVertexValues[ tidWithinCTA ];
				outboxVertices[ positionToWrite ] = fetchedVertexValues[ tidWithinCTA ];
				outboxIndices[ positionToWrite ] = multiDeviceVertexID;
			}

		}

	}
	if( InterDeviceCommunication == VR ) {

		const bool updated = update_condition( fetchedVertexValues + tidWithinCTA, VertexValue + multiDeviceVertexID  );
		if( __any( updated ) ) {
			const bool remoteNeeded = ( fetchedEdgesIndicesInitial[ tidWithinCTA ] & 0x80000000 );
			const uint warpBallot = __ballot( updated & remoteNeeded );
			const int nUpdates = __popc( warpBallot );

			if( nUpdates > 0 ) {
				int indexPosition;
				if( laneID == 0 ) {
					(*devFinished) = 1;
					indexPosition = atomicAdd( movingIndex, nUpdates );
				}
				indexPosition = __shfl( indexPosition, 0 );
				uint laneMask; asm( "mov.u32 %0, %lanemask_lt;" : "=r"(laneMask) );	//const uint laneMask =  0xFFFFFFFF >> ( WARP_SIZE - laneID ) ;
				const int positionToWrite = indexPosition + __popc( warpBallot & laneMask );
				if( updated ) {
					VertexValue[ multiDeviceVertexID ] = fetchedVertexValues[ tidWithinCTA ];
					if( remoteNeeded ) {
						outboxVertices[ positionToWrite ] = fetchedVertexValues[ tidWithinCTA ];
						outboxIndices[ positionToWrite ] = multiDeviceVertexID;
					}
				}
			}
			else {
				if( laneID == 0 )
					(*devFinished) = 1;
				if( updated )
					VertexValue[ multiDeviceVertexID ] = fetchedVertexValues[ tidWithinCTA ];
			}


		}

	}

}

// The main graph processing CUDA kernel for a single device.
__global__ void dev_Warp_Segmentation_single_gpu(
		const uint nVerticesToProcess,
		const uint* vertexIndices,
		const uint* edgesIndices,
		Vertex* VertexValue,
		const Edge* EdgeValue,
		const Vertex_static* VertexValueStatic,
		int* devFinished
		) {

	__shared__ Vertex fetchedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	__shared__ 	Vertex locallyComputedVertexValues[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];
	volatile __shared__ uint fetchedEdgesIndices[ COMPILE_TIME_DETERMINED_BLOCK_SIZE ];

	const uint tidWithinCTA = threadIdx.x;
	uint laneID; asm( "mov.u32 %0, %laneid;" : "=r"(laneID) );	//const uint laneID = tidWithinCTA & ( WARP_SIZE - 1 );
	const uint vertexID = tidWithinCTA + blockIdx.x * blockDim.x;
	if( vertexID >= nVerticesToProcess )
		return;

	// Initialize vertices.
	init_compute( fetchedVertexValues + tidWithinCTA, VertexValue + vertexID );
	const uint warpGlobalVertexOffset = vertexID & ( ~ ( WARP_SIZE - 1 ) );
	fetchedEdgesIndices[ tidWithinCTA ] = edgesIndices[ vertexID ];
	const uint endEdgeIdx = edgesIndices[ warpGlobalVertexOffset + WARP_SIZE ];
	const uint warpOffsetWithinCTA = tidWithinCTA & ( ~ ( WARP_SIZE - 1 ) );
	const uint startEdgeIdx = fetchedEdgesIndices[ warpOffsetWithinCTA ];	// first place in the warp specific shared memory.

	// Compute and reduce using source vertices.
	for( uint currentEdgeIdx = startEdgeIdx + laneID;
			currentEdgeIdx < endEdgeIdx;
			currentEdgeIdx += WARP_SIZE ) {

		const uint targetVertexIndex = vertexIndices[ currentEdgeIdx ];
		const Vertex srcValue = VertexValue[ targetVertexIndex ];

		const uint belongingVertexIdx = find_belonging_vertex_index_inside_warp(
				fetchedEdgesIndices + warpOffsetWithinCTA,
				currentEdgeIdx );

		// reduction parameters. Move right before computation to cover memory access latency.
		const uint nZerosLowerThanPositionInSegmentInclusive = min( laneID, currentEdgeIdx - fetchedEdgesIndices[ warpOffsetWithinCTA + belongingVertexIdx ] );
		const uint nZerosHigherThanPositionInSegment = min( ((belongingVertexIdx==31)?endEdgeIdx:fetchedEdgesIndices[ warpOffsetWithinCTA + belongingVertexIdx + 1 ]) - currentEdgeIdx, WARP_SIZE - laneID );
		const uint segmentSize = nZerosHigherThanPositionInSegment + nZerosLowerThanPositionInSegmentInclusive;
		uint iii = 	( segmentSize > 16 ) ? 32 :
					( segmentSize >  8 ) ? 16 :
					( segmentSize >  4 ) ?  8 :
					( segmentSize >  2 ) ?  4 :
					segmentSize;

		// compute local
		compute_local(
				srcValue,
				VertexValueStatic + targetVertexIndex,
				EdgeValue + currentEdgeIdx,
				locallyComputedVertexValues + tidWithinCTA,
				fetchedVertexValues + warpOffsetWithinCTA + belongingVertexIdx );

		// first reduction iteration.
		if( iii != 1 ) {
			uint jjj = iii >> 1;
			if( ( nZerosLowerThanPositionInSegmentInclusive + jjj ) < segmentSize )
				compute_reduce( locallyComputedVertexValues + tidWithinCTA, locallyComputedVertexValues + tidWithinCTA + jjj );
			iii = jjj;
		}

		// rest of iterations.
		while( iii != 1 ) {
			const uint jjj = iii >> 1;
			if( nZerosLowerThanPositionInSegmentInclusive < jjj )
				compute_reduce( locallyComputedVertexValues + tidWithinCTA, locallyComputedVertexValues + tidWithinCTA + jjj );
			iii = jjj;
		}

		// final reduction by the first lane in the segment.
		if( nZerosLowerThanPositionInSegmentInclusive == 0 ) {
			compute_reduce( fetchedVertexValues + warpOffsetWithinCTA + belongingVertexIdx, locallyComputedVertexValues + tidWithinCTA );
		}

	}

	// Update vertices.
	if( update_condition( fetchedVertexValues + tidWithinCTA, VertexValue + vertexID  ) ) {
		(*devFinished) = 1;
		VertexValue[ vertexID ] = fetchedVertexValues[ tidWithinCTA ];
	}

}


// Host function that launches the main kernel from the host side.
void cuda_kernels::process_graph(
		const uint nVerticesToProcess,
		const uint* vertexIndices,
		const uint* edgesIndices,
		Vertex* VertexValue,
		const Edge* EdgeValue,
		const Vertex_static* VertexValueStatic,
		int* devFinished,
		cudaStream_t sss,
		const uint nDevices,
		const uint vertexOffset,
		const uint edgeOffset,
		uint* movingIndex,
		uint* outboxIndices,
		Vertex* outboxVertices,
		const inter_device_comm InterDeviceCommunication
		) {

	const uint bDim = COMPILE_TIME_DETERMINED_BLOCK_SIZE;
	const uint gDim = std::ceil( static_cast<float>( nVerticesToProcess ) / COMPILE_TIME_DETERMINED_BLOCK_SIZE );

	// Single-GPU kernels.
	if( nDevices == 1 )
		dev_Warp_Segmentation_single_gpu            <<<gDim, bDim, 0, sss>>> ( nVerticesToProcess, vertexIndices, edgesIndices, VertexValue, EdgeValue, VertexValueStatic, devFinished );

	// Multi-GPU with VR.
	if( nDevices > 1 && InterDeviceCommunication == VR )
		dev_Warp_Segmentation_multi_gpu <VR>        <<<gDim, bDim, 0, sss>>> ( nVerticesToProcess, vertexIndices, edgesIndices, VertexValue, EdgeValue, VertexValueStatic, devFinished, vertexOffset, edgeOffset, movingIndex, outboxIndices, outboxVertices);

	// Multi-GPU with VRONLINE.
	if( nDevices > 1 && InterDeviceCommunication == VRONLINE )
		dev_Warp_Segmentation_multi_gpu <VRONLINE>  <<<gDim, bDim, 0, sss>>> ( nVerticesToProcess, vertexIndices, edgesIndices, VertexValue, EdgeValue, VertexValueStatic, devFinished, vertexOffset, edgeOffset, movingIndex, outboxIndices, outboxVertices);

	// Multi-GPU with MS.
	if( nDevices > 1 && InterDeviceCommunication == MS )
		dev_Warp_Segmentation_multi_gpu <MS>        <<<gDim, bDim, 0, sss>>> ( nVerticesToProcess, vertexIndices, edgesIndices, VertexValue, EdgeValue, VertexValueStatic, devFinished, vertexOffset, edgeOffset );

	// Multi-GPU with ALL.
	if( nDevices > 1 && InterDeviceCommunication == ALL )
		dev_Warp_Segmentation_multi_gpu <ALL>       <<<gDim, bDim, 0, sss>>> ( nVerticesToProcess, vertexIndices, edgesIndices, VertexValue, EdgeValue, VertexValueStatic, devFinished, vertexOffset, edgeOffset );

	CUDAErrorCheck( cudaPeekAtLastError() );

}


/*
 * cuda_device.cuh
 *
 *  Created on: Aug 16, 2015
 *      Author: farzad
 */

#ifndef CUDA_DEVICE_CUH_
#define CUDA_DEVICE_CUH_

#include "CUDAErrorCheck.cuh"
#include "cuda_buffers.cuh"
#include "globals.hpp"


template <typename idxT, typename vT, class eT, class vStaticT>
class cuda_device{

public:
	// Variables to mark boundary vertices.
	std::vector< idxT > vertexIndicesToMoveVec;	// Used only in MS.
	uint nVerticesToSend, nVerticesToReceive;
	unsigned long long globalMemCapacity;

	// Device-specific CUDA stream.
	cudaStream_t devStream;

	// Necessary device and host CSR buffers.
	device_buffer< vT > vertexValue;
	device_buffer< idxT > edgesIndices;
	device_buffer< idxT > vertexIndices;
	device_buffer< eT > EdgeValue;
	device_buffer< vStaticT > VertexValueStatic;
	host_pinned_buffer<int> finishedHost;
	device_buffer<int> finished;


	// Outbox buffers. They will be used in multi-GPU scenarios for MS and VR.
	uint vertexOffset, edgeOffset, nDedicatedVertices, nDedicatedEdges;
	device_buffer<uint> outboxTop;
	device_buffer< idxT > outboxIndices;
	device_buffer< vT > outboxVertices;

	device_buffer< idxT > inboxIndices_odd;
	device_buffer< vT > inboxVertices_odd;
	host_pinned_buffer<uint> inboxTop_odd;
	device_buffer< idxT > inboxIndices_even;
	device_buffer< vT > inboxVertices_even;
	host_pinned_buffer<uint> inboxTop_even;

	// MS Temporary host buffer.
	host_pinned_buffer< idxT > tmpHostIndices;


	cuda_device():
		vertexIndicesToMoveVec( 0 ), nVerticesToSend( 0 ), nVerticesToReceive( 0 ),
		vertexOffset( 0 ), edgeOffset( 0 ), nDedicatedVertices( 0 ), nDedicatedEdges( 0 )
	{}
	void create_device_stream(){
		CUDAErrorCheck( cudaStreamCreate( &devStream ) );
	}
	void destroy_device_stream(){
		CUDAErrorCheck( cudaStreamDestroy( devStream ) );
	}

	// Essential CSR buffers.
	void allocate_CSR_buffers( const uint nVertices ) {

		vertexValue.alloc( nVertices );	// A full version of vertices for every device.
		edgesIndices.alloc( nDedicatedVertices + 1 );
		vertexIndices.alloc( nDedicatedEdges );
		if( sizeof(eT) > 1 ) EdgeValue.alloc( nDedicatedEdges );
		if( sizeof(vStaticT) > 1 ) VertexValueStatic.alloc( nVertices );
		finished.alloc( 1 );
		finishedHost.alloc( 1 );

	}

	void allocate_box_buffers(
			const inter_device_comm InterDeviceCommunication,
			const bool HaveHostHub,
			const uint totalNumVertices
			) {

		if( InterDeviceCommunication == VR || InterDeviceCommunication == VRONLINE ) {
			inboxTop_odd.alloc( 1 );
			inboxTop_odd.at( 0 ) = 0;
			inboxTop_even.alloc( 1 );
			inboxTop_even.at( 0 ) = 0;
			outboxTop.alloc( 1 );
			const inter_device_comm met = InterDeviceCommunication;
			outboxIndices.alloc( ( met == VR ) ? nVerticesToSend : totalNumVertices );
			outboxVertices.alloc( ( met == VR ) ? nVerticesToSend : totalNumVertices );
			const uint nVerticesFromOtherDevices = totalNumVertices - nDedicatedVertices;
			if( !HaveHostHub ) { // If we don't have host as the hub, we need inbox buffers in each device.
				inboxIndices_odd.alloc( ( met == VR ) ? nVerticesToReceive : nVerticesFromOtherDevices );
				inboxVertices_odd.alloc( ( met == VR ) ? nVerticesToReceive : nVerticesFromOtherDevices );
				inboxIndices_even.alloc( ( met == VR ) ? nVerticesToReceive : nVerticesFromOtherDevices );
				inboxVertices_even.alloc( ( met == VR ) ? nVerticesToReceive : nVerticesFromOtherDevices );
			}
		}
		if( InterDeviceCommunication == MS ) {
			outboxIndices.alloc( nVerticesToSend );
			outboxVertices.alloc( nVerticesToSend );
			inboxVertices_odd.alloc( nVerticesToReceive );	// Just an inbox buffer to receive vertices from other devices.
			inboxIndices_odd.alloc( nVerticesToReceive );

			tmpHostIndices.alloc( nVerticesToSend );
			for( uint iii = 0; iii < vertexIndicesToMoveVec.size(); ++iii )
				tmpHostIndices.at( iii ) = vertexIndicesToMoveVec.at( iii );
			vertexIndicesToMoveVec.clear();	// Free-up the the memory for the vector since we don't need it anymore.
		}

	}

};


#endif /* CUDA_DEVICE_CUH_ */

#ifndef CUDA_KERNELS_CUH_
#define CUDA_KERNELS_CUH_

#include "user_specified_codes/user_specified_structures.h"
#include "utils/globals.hpp"

namespace cuda_kernels{

void unload_inbox(
		Vertex* VertexValue,
		const uint loadSize,
		const uint* inboxIndices,
		const Vertex* inboxVertices,
		cudaStream_t sss,
		const uint offset = 0
		);

void fill_outbox(
		const Vertex* VertexValue,
		const uint loadSize,
		const uint* outboxIndices,
		Vertex* outboxVertices,
		cudaStream_t sss
		);

void distribute_outbox(
		const uint maxLoadSize,
		const uint* loadSizePtr,
		const uint* srcIndices,
		const Vertex* srcVertices,
		uint* dstIndices,
		Vertex* dstVertices,
		cudaStream_t sss,
		uint dstOffset = 0
		);


void process_graph(
		const uint nVerticesToProcess,
		const uint* vertexIndices,
		const uint* edgesIndices,
		Vertex* VertexValue,
		const Edge* EdgeValue,
		const Vertex_static* VertexValueStatic,
		int* devFinished,
		cudaStream_t sss,
		const uint nDevices = 1,
		const uint vertexOffset = 0,
		const uint edgeOffset = 0,
		uint* movingIndex = NULL,
		uint* outboxIndices = NULL,
		Vertex* outboxVertices = NULL,
		const inter_device_comm InterDeviceCommunication = VR
		);

}


#endif /* CUDA_KERNELS_CUH_ */

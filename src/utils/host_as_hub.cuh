#ifndef HOST_AS_HUB_CUH_
#define HOST_AS_HUB_CUH_

#include "cuda_buffers.cuh"

template <typename idxT, typename vT>
class host_as_hub{

public:
	host_pinned_buffer<vT> vertexValue_odd;
	host_pinned_buffer<vT> vertexValue_even;
	host_pinned_buffer<idxT> vertexIndices_odd;
	host_pinned_buffer<idxT> vertexIndices_even;

	void createHub( const std::size_t nDevices, const std::size_t totalLength ) {
		vertexValue_odd.alloc( totalLength );
		vertexIndices_odd.alloc( totalLength );
		vertexValue_even.alloc( totalLength );
		vertexIndices_even.alloc( totalLength );
	}

};


#endif /* HOST_AS_HUB_CUH_ */

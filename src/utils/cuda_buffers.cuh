#ifndef CUDA_BUFFERS_CUH
#define CUDA_BUFFERS_CUH

#include <stdexcept>

#include "CUDAErrorCheck.cuh"

template <typename T>
class device_buffer;

template <typename T>
class host_pinned_buffer;



template <typename T>
class host_pinned_buffer{
private:
	T* ptr;
	size_t nElems;
	void construct(size_t n){
		CUDAErrorCheck( cudaHostAlloc( (void**)&ptr, n*sizeof(T), cudaHostAllocPortable ) );
		nElems = n;
	}
public:
	host_pinned_buffer(){
		nElems = 0;
		ptr = NULL;
	}
	host_pinned_buffer(size_t n){
		construct(n);
	}
	~host_pinned_buffer(){
		if( nElems!=0 )
			CUDAErrorCheck( cudaFreeHost( ptr ) );
	}
	void alloc(size_t n){
		if( nElems==0 )
			construct(n);
	}
	void free(){
		if( nElems!=0 ) {
			nElems = 0;
			CUDAErrorCheck( cudaFreeHost( ptr ) );
		}
	}
	T& at(size_t index){
		if( index >= nElems )
			throw std::runtime_error( "The referred element does not exist in the buffer." );
		return ptr[index];
	}
	T& operator[](size_t index){	// is faster compared to 'at'.
		return ptr[index];
	}
	T* get_ptr(){
		return ptr;
	}
	size_t size(){
		return nElems;
	}
	size_t sizeInBytes(){
		return nElems*sizeof(T);
	}
	host_pinned_buffer<T>& copy_all( device_buffer<T>& srcDevBuffer, cudaStream_t sss = 0 ) {
	    if( nElems == 0 ) {
	    	construct( srcDevBuffer.size() );
	    	CUDAErrorCheck( cudaMemcpyAsync( ptr, srcDevBuffer.get_ptr(), srcDevBuffer.sizeInBytes(), cudaMemcpyDeviceToHost, sss ) );
	    }
	    else {
	    	size_t copySize = ( srcDevBuffer.sizeInBytes() < this->sizeInBytes() ) ? srcDevBuffer.sizeInBytes() : this->sizeInBytes();
	    	CUDAErrorCheck( cudaMemcpyAsync( ptr, srcDevBuffer.get_ptr(), copySize, cudaMemcpyDeviceToHost, sss ) );
	    }
	    return *this;
	}
	host_pinned_buffer<T>& operator=( device_buffer<T>& srcDevBuffer )
	{
	    copy_all( srcDevBuffer );
	    return *this;
	}

};

template <typename T>
class device_buffer{
private:
	T* ptr;
	size_t nElems;
	void construct(size_t n) {
		CUDAErrorCheck( cudaMalloc( (void**)&ptr, n*sizeof(T) ) );
		nElems = n;
	}
public:
	device_buffer():
		nElems(0), ptr(NULL)
	{}
	device_buffer(size_t n){
		construct(n);
	}
	~device_buffer(){
		if( nElems!=0 )
			CUDAErrorCheck( cudaFree( ptr ) );
	}
	void alloc(size_t n){
		if( nElems==0 )
			construct(n);
	}
	void free(){
		if( nElems!=0 ) {
			nElems = 0;
			CUDAErrorCheck( cudaFree( ptr ) );
		}
	}
	T* get_ptr(){
		return ptr;
	}
	size_t size(){
		return nElems;
	}
	size_t sizeInBytes(){
		return nElems*sizeof(T);
	}

	device_buffer<T>& copy_all( host_pinned_buffer<T>& srcHostBuffer, cudaStream_t sss = 0 ) {
	    if( nElems == 0 ) {
	    	construct( srcHostBuffer.size() );
	    	CUDAErrorCheck( cudaMemcpyAsync( ptr, srcHostBuffer.get_ptr(), srcHostBuffer.sizeInBytes(), cudaMemcpyHostToDevice, sss ) );
	    }
	    else {
	    	size_t copySize = ( srcHostBuffer.sizeInBytes() < this->sizeInBytes() ) ? srcHostBuffer.sizeInBytes() : this->sizeInBytes();
	    	CUDAErrorCheck( cudaMemcpyAsync( ptr, srcHostBuffer.get_ptr(), copySize, cudaMemcpyHostToDevice, sss ) );
	    }
	    return *this;
	}
	device_buffer<T>& operator=( host_pinned_buffer<T>& srcHostBuffer )
	{
	    copy_all( srcHostBuffer );
	    return *this;
	}

	device_buffer<T>& copy_section( host_pinned_buffer<T>& srcHostBuffer, std::size_t offset, std::size_t nSrcElems, cudaStream_t sss = 0, std::size_t dstOffset = 0 ) {
		 if( nElems == 0 ) {
			 construct( nSrcElems );
			 CUDAErrorCheck( cudaMemcpyAsync( ptr + dstOffset, srcHostBuffer.get_ptr() + offset, nSrcElems * sizeof(T), cudaMemcpyHostToDevice, sss ) );
		 }
		 else {
			 size_t copySize = ( nSrcElems * sizeof(T) < this->sizeInBytes() ) ? nSrcElems * sizeof(T) : this->sizeInBytes();
			 CUDAErrorCheck( cudaMemcpyAsync( ptr + dstOffset, srcHostBuffer.get_ptr() + offset, copySize, cudaMemcpyHostToDevice, sss ) );
		 }
		return *this;
	}
};



#endif	//	CUDA_BUFFERS_CUH

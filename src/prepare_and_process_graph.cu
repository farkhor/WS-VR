#include <iostream>

#include "prepare_and_process_graph.cuh"
#include "utils/CUDAErrorCheck.cuh"
#include "utils/cuda_buffers.cuh"
#include "utils/cuda_device.cuh"
#include "utils/host_as_hub.cuh"
#include "utils/timer.h"
#include "iterative_procedure.cuh"
#include "user_specified_codes/user_specified_pre_and_post_processing_functions.hpp"

void prepare_and_process_graph(
		std::vector<initial_vertex>* initGraph,
		const uint nEdges,
		std::ofstream& outputFile,
		const std::vector<unsigned int>* indicesRange,
		const int nDevices,
		const int singleDeviceID,
		const inter_device_comm InterDeviceCommunication
		) {

	// Get the number of vertices.
	const uint nVertices = initGraph->size();

	// Check availability of selected devices.
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		CUDAErrorCheck( cudaFree( 0 ) );
	}

	/*********************************************************************************
	 * ALLOCATE HOST BUFFERS AND PUT VERTICES INTO HOST BUFFERS OF CSR REPRESENTATION.
	 ********************************************************************************/

	// Allocate host buffers.
	host_pinned_buffer<Vertex> vertexValue( nVertices );
	host_pinned_buffer<uint> edgesIndices( nVertices + 1 );
	edgesIndices.at(0) = 0;
	host_pinned_buffer<uint> vertexIndices( nEdges );
	host_pinned_buffer<Edge> EdgeValue;
	if( sizeof(Edge) > 1 ) EdgeValue.alloc( nEdges );
	host_pinned_buffer<Vertex_static> VertexValueStatic;
	if( sizeof(Vertex_static) > 1 ) VertexValueStatic.alloc( nVertices );

	// Put vertices into host buffer CSR form.
	for( uint vIdx = 0; vIdx < nVertices; ++vIdx ) {
		initial_vertex& vvv = initGraph->at(vIdx);
		vertexValue[ vIdx ] = vvv.vertexValue;
		if( sizeof(Vertex_static) > 1 ) VertexValueStatic[ vIdx ] = vvv.VertexValueStatic;
		uint nNbrs = vvv.nbrs.size();
		uint edgeIdxOffset = edgesIndices[ vIdx ];
		for( uint nbrIdx = 0; nbrIdx < nNbrs; ++nbrIdx ) {
			neighbor& nbr = vvv.nbrs.at( nbrIdx );
			vertexIndices[ edgeIdxOffset + nbrIdx ] = nbr.srcIndex;
			if( sizeof(Edge) > 1 ) EdgeValue[ edgeIdxOffset + nbrIdx ] = nbr.edgeValue;
		}
		edgesIndices[ vIdx + 1 ] = edgeIdxOffset + nNbrs;
	}

	// Free-up some occupied memory.
	initGraph->resize( 0 );

	/*********************************************************************************
	 * INITIALIZE DEVICE GRAPH-RELATED PARAMETERS.
	 ********************************************************************************/

	// The set of devices to be used.
	std::vector< cuda_device< uint, Vertex, Edge, Vertex_static > > computingDevices( nDevices );
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );
		devInHand.create_device_stream();
		// Find out the number of vertices and edges of the graph portion assigned to this device.
		devInHand.vertexOffset = indicesRange->at( devID );
		devInHand.nDedicatedVertices = indicesRange->at( devID + 1 ) - devInHand.vertexOffset;
		devInHand.edgeOffset = edgesIndices.at( indicesRange->at( devID ) );
		devInHand.nDedicatedEdges = edgesIndices.at( indicesRange->at( devID + 1 ) ) - devInHand.edgeOffset;
		// Query the device's global memory capacity.
		cudaDeviceProp devProp;
		CUDAErrorCheck( cudaGetDeviceProperties(&devProp, ( nDevices == 1 ) ? singleDeviceID : devID ) );
		devInHand.globalMemCapacity = devProp.totalGlobalMem;
	}

	/*********************************************************************************
	 * RECOGNIZE BOUNDARY VERTICES.
	 ********************************************************************************/

	// Mark the vertices passing partition boundaries.
	uint totalNumVerticesMarked = 0;
	if( nDevices > 1 ) {

		if( InterDeviceCommunication != ALL ) {

			// A vector that will hold true for a vertex's corresponding index if it is referred to by other devices.
			std::vector< bool > vertexMarker( nVertices, false );

			// Mark the vertices that are referred to by other devices.
			for( uint devID = 0; devID < nDevices; ++devID ) {
				for( uint edgeIdx = edgesIndices.at( indicesRange->at( devID )  );
						edgeIdx < edgesIndices.at( indicesRange->at( devID + 1 ) );
						++edgeIdx ) {
					const uint nbrIdx = vertexIndices.at( edgeIdx );
					uint nbrDevID;
					for( nbrDevID = 0; nbrDevID < nDevices; ++nbrDevID )
						if( ( nbrIdx >= indicesRange->at( nbrDevID ) ) && ( nbrIdx < indicesRange->at( nbrDevID + 1 ) ) )
							break;
					if( nbrDevID != devID ) {
						vertexMarker.at( nbrIdx ) = true;
						++totalNumVerticesMarked;
					}
				}
			}

			// Realize marked vertices for both methods.
			if( InterDeviceCommunication == MS ) {	// MS.
				for( uint devID = 0; devID < nDevices; ++devID ) {
					for( uint vIdx = indicesRange->at( devID );
							vIdx < indicesRange->at( devID + 1 );
							++vIdx ) {
						if( vertexMarker.at( vIdx ) )
							computingDevices.at( devID ).vertexIndicesToMoveVec.push_back( vIdx );
					}
					computingDevices.at( devID ).nVerticesToSend = computingDevices.at( devID ).vertexIndicesToMoveVec.size();
				}

			}
			else {	// VR or VRONLINE.
				for( uint devID = 0; devID < nDevices; ++devID )
					for( uint vIdx = indicesRange->at( devID );
							vIdx < indicesRange->at( devID + 1 );
							++vIdx )
						if( vertexMarker.at( vIdx ) ) {
							if( InterDeviceCommunication == VR )	// Mark the MSB for Offline part of VR.
								edgesIndices[ vIdx ] |= 0x80000000;
							computingDevices.at( devID ).nVerticesToSend = computingDevices.at( devID ).nVerticesToSend + 1;
						}
			}

		} else { // In case of ALL.
			for( uint devID = 0; devID < nDevices; ++devID )
				computingDevices.at( devID ).nVerticesToSend = computingDevices.at( devID ).nDedicatedVertices;
		}

		// Calculate the number of vertices to receive for each device by summing up the number of vertices other devices have to transfer.
		for( uint devID = 0; devID < nDevices; ++devID ) {
			for( uint trgtDevID = 0; trgtDevID < nDevices; ++trgtDevID )
				if( devID != trgtDevID )
					computingDevices.at( devID ).nVerticesToReceive += computingDevices.at( trgtDevID ).nVerticesToSend;
		}

	}


	/*********************************************************************************
	 * FRAMEWORK DECIDES WHERE TO PUT THE BUFFERS VIA AN ESTIMATION.
	 ********************************************************************************/

	bool HaveHostHub = ( nDevices > 1 ) && ( InterDeviceCommunication == VR || InterDeviceCommunication == VRONLINE ); // For more than 2 devices, use the host as the hub.
	if( nDevices == 2 && ( InterDeviceCommunication == VR || InterDeviceCommunication == VRONLINE ) ) {
		HaveHostHub = false;
		for( uint devID = 0; devID < nDevices; ++devID ) {
			cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );
			unsigned long long requiredSizes = nVertices * ( sizeof(Vertex) + sizeof(Vertex_static) ) +	// Aware: the size of an empty structure can be 1.
					devInHand.nDedicatedEdges * ( sizeof(Edge) + sizeof(uint) ) +
					( devInHand.nDedicatedVertices + 1 ) * sizeof(uint) +
					devInHand.nVerticesToSend * ( sizeof(Vertex) + sizeof(uint) );	// Outbox
			unsigned long long cap = devInHand.globalMemCapacity - 50000000; // We set 50MB as the safety zone.
			if( requiredSizes > cap )
				throw std::runtime_error( "The graph seems to be too big for the device." );
			if( ( requiredSizes + devInHand.nVerticesToReceive * 2 * ( sizeof(Vertex) + sizeof(uint) ) ) > cap ) {	// Consider the inboxes place.
				HaveHostHub = true;
				break;
			}
		}
		if( HaveHostHub == false ) {	// Check if peer-access is possible in a two-GPU configuration.
			for( uint devID = 0; devID < nDevices; ++devID ) {
				CUDAErrorCheck( cudaSetDevice( devID ) );
				for( uint targetDevID = 0; targetDevID < nDevices; ++targetDevID ) {
					if( targetDevID != devID ) {
						int canAccessPeer = 0;
						CUDAErrorCheck( cudaDeviceCanAccessPeer( &canAccessPeer, devID, targetDevID ) );
						if( canAccessPeer == 0 ) {
							HaveHostHub = true;
							break;
						} else
							CUDAErrorCheck( cudaDeviceEnablePeerAccess( targetDevID , 0 ) );
					}
				}
				if( HaveHostHub ) break;
			}
		}

	}

	if( nDevices > 1 && HaveHostHub ) std::cout << "Host-as-hub is acive." << std::endl;

	/*************************************************************************************
	 * DEFINE AND ALLOCATE HOST-AS-HUB.
	 *************************************************************************************/

	host_as_hub< uint, Vertex > hostHub;
	if( ( nDevices > 1 )  && HaveHostHub )	// We need to have a hub in host for VR and VRONLINE.
		if( InterDeviceCommunication == VR )
			hostHub.createHub( nDevices, totalNumVerticesMarked );
		else if( InterDeviceCommunication == VRONLINE )
			hostHub.createHub( nDevices, nVertices );

	/*************************************************************************************
	 * ALLOCATE DEVICE BUFFERS.
	 *************************************************************************************/

	for( uint devID = 0; devID < nDevices; ++devID ) {	// For every device.

		// Set the device first.
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );

		// Allocate buffers.
		devInHand.allocate_CSR_buffers( nVertices );
		if( nDevices > 1 )	// In case of multi-GPU processing.
			devInHand.allocate_box_buffers( InterDeviceCommunication, HaveHostHub, nVertices );

	}

	/*************************************************************************************
	 * COPY FROM THE HOST TO DEVICE(S).
	 *************************************************************************************/

	// First sync with them all.
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		CUDAErrorCheck( cudaDeviceSynchronize() );
	}
	timer::setTime();
	for( uint devID = 0; devID < nDevices; ++devID ) {	// For every device.

		// Set the device first.
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );

		// Copy CSR buffers.
		devInHand.vertexValue.copy_all( vertexValue, devInHand.devStream );
		devInHand.edgesIndices.copy_section( edgesIndices, devInHand.vertexOffset, devInHand.nDedicatedVertices + 1, devInHand.devStream );
		devInHand.vertexIndices.copy_section( vertexIndices, devInHand.edgeOffset, devInHand.nDedicatedEdges, devInHand.devStream );
		if( sizeof(Edge) > 1 ) devInHand.EdgeValue.copy_section( EdgeValue, devInHand.edgeOffset, devInHand.nDedicatedEdges, devInHand.devStream );
		if( sizeof(Vertex_static) > 1 ) devInHand.VertexValueStatic.copy_all( VertexValueStatic, devInHand.devStream );

		// If MS, fill up the box indices required.
		if( nDevices > 1 && InterDeviceCommunication == MS ) {
			std::size_t iii = 0;
			for( uint trgtDevID = 0; trgtDevID < nDevices; ++trgtDevID ) {
				if( trgtDevID != devID ) {
					devInHand.inboxIndices_odd.copy_section( computingDevices.at( trgtDevID ).tmpHostIndices, 0, computingDevices.at( trgtDevID ).tmpHostIndices.size(), devInHand.devStream, iii );
					iii += computingDevices.at( trgtDevID ).tmpHostIndices.size();
				}
				else {
					devInHand.outboxIndices.copy_all( devInHand.tmpHostIndices, devInHand.devStream );
				}
			}

		}

	}
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		CUDAErrorCheck( cudaDeviceSynchronize() );
	}

	const float H2D_copy_time = timer::getTime();
	std::cout << "Copying data to devices took " << H2D_copy_time << " (ms).\n";

	// Free MS temporary memory.
	if( nDevices > 1 && InterDeviceCommunication == MS ) {
		for( uint devID = 0; devID < nDevices; ++devID ) {	// For every device.
			CUDAErrorCheck( cudaSetDevice( devID ) );
			computingDevices.at( devID ).tmpHostIndices.free();
		}
	}

	/*************************************************************************************
	 * START PROCESSING THE GRAPH.
	 *************************************************************************************/

	// First sync with them all.
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		CUDAErrorCheck( cudaDeviceSynchronize() );
	}
	// Set the timer and start.
	timer::setTime();
	unsigned int IterationCounter = iterative_procedure(
			nDevices,
			singleDeviceID,
			computingDevices,
			hostHub,
			HaveHostHub,
			InterDeviceCommunication
			);

	const float processing_time = timer::getTime();
	std::cout << "Processing finished in " << processing_time << " (ms).\n";
	std::cout << "Performed " << IterationCounter << " iterations in total.\n";

	/*************************************************************************************
	 * COPY BACK THE RESULTS BACK TO THE HOST SIDE.
	 *************************************************************************************/

	timer::setTime();
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );
		CUDAErrorCheck( cudaMemcpyAsync(
				vertexValue.get_ptr() + devInHand.vertexOffset,
				devInHand.vertexValue.get_ptr() + devInHand.vertexOffset,
				devInHand.nDedicatedVertices * sizeof(Vertex),
				cudaMemcpyDeviceToHost ) );
	}
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		CUDAErrorCheck( cudaDeviceSynchronize() );
	}
	const float D2H_copy_time = timer::getTime();
	std::cout << "Copying final vertex values back from the device to the host finished in " << D2H_copy_time << " (ms).\n";
	// Destroy the streams.
	for( uint devID = 0; devID < nDevices; ++devID ) {
		CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
		computingDevices.at( devID ).destroy_device_stream();
	}

	/*************************************************************************************
	 * PERFORM USER-SPECIFIED OUTPUT FUNCTION.
	 *************************************************************************************/

	// Print the output vertex values to the file.
	for( uint vvv = 0; vvv < nVertices; ++vvv )
		print_vertex_output(
			vvv,
			vertexValue[ vvv ],
			outputFile
			);

}

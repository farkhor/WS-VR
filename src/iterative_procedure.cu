#include "iterative_procedure.cuh"
#include "utils/CUDAErrorCheck.cuh"
#include "utils/cuda_buffers.cuh"
#include "cuda_kernels.cuh"


unsigned int iterative_procedure(
		const unsigned int nDevices,	// Number of devices involved.
		const int singleDeviceID,
		std::vector< cuda_device< uint, Vertex, Edge, Vertex_static > >& computingDevices,	// The devices and the buffers each hold.
		host_as_hub< uint, Vertex >& hostHub,												// The host buffers that will act as the hub.
		const bool HaveHostHub,																// True if we have the host-as-hub.
		const inter_device_comm comMeth									// The method for devices to communicate to each other.
		) {


	// Iteratively process with a do-while loop.
	unsigned int iterationCounter( 0 );
	int finished( 0 );
	do{

		/**************************************************************************************************
		 * SINGLE-GPU SCENARIO.
		 **************************************************************************************************/

		if( nDevices == 1 ) {
			const uint devID = 0;
			CUDAErrorCheck( cudaSetDevice( singleDeviceID ) );
			cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );
			finished = 0;
			devInHand.finishedHost.at( 0 ) = 0;
			devInHand.finished.copy_all( devInHand.finishedHost, devInHand.devStream );
			cuda_kernels::process_graph(
					devInHand.nDedicatedVertices,
					devInHand.vertexIndices.get_ptr(),
					devInHand.edgesIndices.get_ptr(),
					devInHand.vertexValue.get_ptr(),
					devInHand.EdgeValue.get_ptr(),
					devInHand.VertexValueStatic.get_ptr(),
					devInHand.finished.get_ptr(),
					devInHand.devStream
					);
			devInHand.finishedHost.copy_all( devInHand.finished, devInHand.devStream );

		}

		/**************************************************************************************************
		 * MULTI-GPU SCENARIO WITH VR/VRONLINE.
		 **************************************************************************************************/

		if( nDevices > 1 && ( comMeth == VR || comMeth == VRONLINE ) ) {

			const bool oddIter = ( iterationCounter & 1 ) == 0 ;

			uint devInHandHostInboxOffset = 0;
			uint devInHandDevInboxOffset = 0;
			for( uint devID = 0; devID < nDevices; ++devID ) {
				CUDAErrorCheck( cudaSetDevice( devID ) );
				cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );

				devInHand.finishedHost.at( 0 ) = finished = 0;
				devInHand.finished.copy_all( devInHand.finishedHost, devInHand.devStream );

				// Initialize the outbox top.
				if( oddIter ) 	devInHand.inboxTop_odd.at( 0 ) = 0;
				else			devInHand.inboxTop_even.at( 0 ) = 0;
				devInHand.outboxTop.copy_all( oddIter ? devInHand.inboxTop_odd : devInHand.inboxTop_even, devInHand.devStream );

				// Unload the inbox first.
				if( iterationCounter != 0 ) {

					uint hostInboxOffset = 0, devInboxOffset = 0;
					for( uint targetDevID = 0; targetDevID < nDevices; ++targetDevID ) {
						if( targetDevID != devID ) {
							if( HaveHostHub ) {	// Have the host as the hub.
								cuda_kernels::unload_inbox(
									devInHand.vertexValue.get_ptr(),
									oddIter ? computingDevices.at( targetDevID ).inboxTop_even.at( 0 ) : computingDevices.at( targetDevID ).inboxTop_odd.at( 0 ),
									oddIter ? hostHub.vertexIndices_even.get_ptr() : hostHub.vertexIndices_odd.get_ptr(),
									oddIter ? hostHub.vertexValue_even.get_ptr() : hostHub.vertexValue_odd.get_ptr(),
									devInHand.devStream,
									hostInboxOffset );
							} else {	// inboxes are inside the device.
								cuda_kernels::unload_inbox(
									devInHand.vertexValue.get_ptr(),
									oddIter ? computingDevices.at( targetDevID ).inboxTop_even.at( 0 ) : computingDevices.at( targetDevID ).inboxTop_odd.at( 0 ),
									oddIter ? devInHand.inboxIndices_even.get_ptr() : devInHand.inboxIndices_odd.get_ptr(),
									oddIter ? devInHand.inboxVertices_even.get_ptr() : devInHand.inboxVertices_odd.get_ptr(),
									devInHand.devStream,
									devInboxOffset );
								devInboxOffset += ( ( comMeth == VR ) ? computingDevices.at( targetDevID ).nVerticesToSend : computingDevices.at( targetDevID ).nDedicatedVertices );
							}
						}
						hostInboxOffset += ( ( comMeth == VR ) ? computingDevices.at( targetDevID ).nVerticesToSend : computingDevices.at( targetDevID ).nDedicatedVertices );
					}
				}

				// Launch the graph processing kernel.
				cuda_kernels::process_graph(
						devInHand.nDedicatedVertices,
						devInHand.vertexIndices.get_ptr(),
						devInHand.edgesIndices.get_ptr(),
						devInHand.vertexValue.get_ptr(),
						devInHand.EdgeValue.get_ptr(),
						devInHand.VertexValueStatic.get_ptr(),
						devInHand.finished.get_ptr(),
						devInHand.devStream,
						nDevices,
						devInHand.vertexOffset,
						devInHand.edgeOffset,
						devInHand.outboxTop.get_ptr(),
						devInHand.outboxIndices.get_ptr(),
						devInHand.outboxVertices.get_ptr(),
						comMeth
						);
				devInHand.finishedHost.copy_all( devInHand.finished, devInHand.devStream );
				if( oddIter ) devInHand.inboxTop_odd.copy_all( devInHand.outboxTop, devInHand.devStream );
				else          devInHand.inboxTop_even.copy_all( devInHand.outboxTop, devInHand.devStream );

				// Distribute/copy the outbox to host/other device's inboxes.
				if( HaveHostHub ) {
					cuda_kernels::distribute_outbox(
							( ( comMeth == VR ) ? devInHand.nVerticesToSend : devInHand.nDedicatedVertices ),
						devInHand.outboxTop.get_ptr(),
						devInHand.outboxIndices.get_ptr(),
						devInHand.outboxVertices.get_ptr(),
						oddIter ? hostHub.vertexIndices_odd.get_ptr() : hostHub.vertexIndices_even.get_ptr(),
						oddIter ? hostHub.vertexValue_odd.get_ptr() : hostHub.vertexValue_even.get_ptr(),
						devInHand.devStream,
						devInHandHostInboxOffset );
					devInHandHostInboxOffset += ( ( comMeth == VR ) ? devInHand.nVerticesToSend : devInHand.nDedicatedVertices );

				} else {	// Two device holding inboxes.
					for( uint targetDevID = 0; targetDevID < nDevices; ++targetDevID )
						if( targetDevID != devID ) {	// Made possible by direct peer memory access.
							cuda_kernels::distribute_outbox(
								( ( comMeth == VR ) ? devInHand.nVerticesToSend : devInHand.nDedicatedVertices ),
								devInHand.outboxTop.get_ptr(),
								devInHand.outboxIndices.get_ptr(),
								devInHand.outboxVertices.get_ptr(),
								oddIter ? computingDevices.at( targetDevID ).inboxIndices_odd.get_ptr() : computingDevices.at( targetDevID ).inboxIndices_even.get_ptr(),
								oddIter ? computingDevices.at( targetDevID ).inboxVertices_odd.get_ptr() : computingDevices.at( targetDevID ).inboxVertices_even.get_ptr(),
								devInHand.devStream,
								devInHandHostInboxOffset );
							devInHandDevInboxOffset += ( ( comMeth == VR ) ? devInHand.nVerticesToSend : devInHand.nDedicatedVertices );
						}
				}

			}	// End of the for loop on the device.

		}	// End of the condition for VR/VRONLINE.

		/**************************************************************************************************
		 * MULTI-GPU SCENARIO WITH MS/ALL.
		 **************************************************************************************************/

		if( nDevices > 1 && ( comMeth == MS || comMeth == ALL ) ) {

			// First round of iterating over devices. We launch kernels at this round.
			for( uint devID = 0; devID < nDevices; ++devID ) {
				CUDAErrorCheck( cudaSetDevice( devID ) );
				cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );
				devInHand.finishedHost.at( 0 ) = finished = 0;
				devInHand.finished.copy_all( devInHand.finishedHost, devInHand.devStream );




				cuda_kernels::process_graph(
						devInHand.nDedicatedVertices,
						devInHand.vertexIndices.get_ptr(),
						devInHand.edgesIndices.get_ptr(),
						devInHand.vertexValue.get_ptr(),
						devInHand.EdgeValue.get_ptr(),
						devInHand.VertexValueStatic.get_ptr(),
						devInHand.finished.get_ptr(),
						devInHand.devStream,
						nDevices,
						devInHand.vertexOffset,
						devInHand.edgeOffset,
						devInHand.outboxTop.get_ptr(),
						devInHand.outboxIndices.get_ptr(),
						devInHand.outboxVertices.get_ptr(),
						comMeth
						);
				devInHand.finishedHost.copy_all( devInHand.finished, devInHand.devStream );

				// For MS the outbox buffers need to be filled via copies.
				if( iterationCounter != 0 && comMeth == MS )	// Outbox must be filled.
						cuda_kernels::fill_outbox(
								devInHand.vertexValue.get_ptr(),
								devInHand.nVerticesToSend,
								devInHand.outboxIndices.get_ptr(),
								devInHand.outboxVertices.get_ptr(),
								devInHand.devStream
								);

			}

			// Sync because we have a single inbox in this case.
			for( uint devID = 0; devID < nDevices; ++devID ) {
				CUDAErrorCheck( cudaSetDevice( devID ) );
				CUDAErrorCheck( cudaDeviceSynchronize() );
			}

			// Second round.
			for( uint devID = 0; devID < nDevices; ++devID ) {
				CUDAErrorCheck( cudaSetDevice( devID ) );
				cuda_device< uint, Vertex, Edge, Vertex_static >& devInHand = computingDevices.at( devID );


				// For ALL, we need to copy this vertex value buffer to other devices' buffers.
				if( iterationCounter != 0 &&comMeth == ALL ) {
					for( uint targetDevID = 0; targetDevID < nDevices; ++targetDevID )
						if( targetDevID != devID )
							CUDAErrorCheck( cudaMemcpyAsync(
								computingDevices.at( targetDevID ).vertexValue.get_ptr() + devInHand.vertexOffset,
								devInHand.vertexValue.get_ptr() + devInHand.vertexOffset,
								devInHand.nDedicatedVertices * sizeof(Vertex),
								cudaMemcpyDeviceToDevice,
								devInHand.devStream ) );
				}


				// For MS
				if( iterationCounter != 0 &&comMeth == MS ) {

					// First copy other outboxes to the device's inbox.
					uint inboxOffset = 0;
					for( uint targetDevID = 0; targetDevID < nDevices; ++targetDevID )
						if( targetDevID != devID ) {
							CUDAErrorCheck( cudaMemcpyAsync(
								devInHand.inboxVertices_odd.get_ptr() + inboxOffset,
								computingDevices.at( targetDevID ).outboxVertices.get_ptr(),
								computingDevices.at( targetDevID ).outboxVertices.sizeInBytes(),
								cudaMemcpyDeviceToDevice,
								devInHand.devStream ) );
							inboxOffset += computingDevices.at( targetDevID ).outboxVertices.size();
						}


					// Then unload the inbox.
					cuda_kernels::unload_inbox(
						devInHand.vertexValue.get_ptr(),
						devInHand.nVerticesToReceive,
						devInHand.inboxIndices_odd.get_ptr(),
						devInHand.inboxVertices_odd.get_ptr(),
						devInHand.devStream
						);

				}	// End of MS.

			}	// End of second round.

		}	// End of multi-GPU scenario with MS/ALL.

		/**************************************************************************************************
		 * FOR ALL SCENARIOS, SYNC WITH THE HOST AT THE END OF AN ITERATION.
		 **************************************************************************************************/

		// Sync with devices at the end of the iteration and apply device-specific 'finished' flag effect.
		for( uint devID = 0; devID < nDevices; ++devID ) {
			CUDAErrorCheck( cudaSetDevice( ( nDevices == 1 ) ? singleDeviceID : devID ) );
			CUDAErrorCheck( cudaDeviceSynchronize() );
			finished |= computingDevices.at( devID ).finishedHost[0];
		}

		++iterationCounter;
	} while( finished == 1 );

	return iterationCounter;
}

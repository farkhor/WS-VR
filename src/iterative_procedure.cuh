#ifndef ITERATIVE_PROCEDURE_CUH_
#define ITERATIVE_PROCEDURE_CUH_

#include <vector>

#include "utils/cuda_device.cuh"
#include "utils/host_as_hub.cuh"
#include "utils/globals.hpp"
#include "user_specified_codes/user_specified_structures.h"


unsigned int iterative_procedure(															// Returning the number of iterations it takes for the graph computation to converge.
		const unsigned int nDevices,														// Number of devices involved.
		const int singleDeviceID,															// The selected device ID if the mode is single GPU.
		std::vector< cuda_device< uint, Vertex, Edge, Vertex_static > >& computingDevices,	// The devices and the buffers each hold.
		host_as_hub< uint, Vertex >& hostHub,												// The host buffers that will act as the hub.
		const bool HaveHostHub,																// True if we have the host-as-hub.
		const inter_device_comm InterDeviceCommunication									// The method for devices to communicate to each other.
		);


#endif /* ITERATIVE_PROCEDURE_CUH_ */

#ifndef PREPARE_AND_PROCESS_CUH_
#define PREPARE_AND_PROCESS_CUH_

#include <vector>
#include <fstream>

#include "utils/init_graph.hpp"
#include "utils/globals.hpp"

void prepare_and_process_graph(
		std::vector<initial_vertex>* initGraph,
		const uint nEdges,
		std::ofstream& outputFile,
		const std::vector<unsigned int>* indicesRange,
		const int nDevices,
		const int singleDeviceID,
		const inter_device_comm InterDeviceCommunication
		);

#endif /* PREPARE_AND_PROCESS_CUH_ */

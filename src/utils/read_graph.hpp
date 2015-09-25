#ifndef READ_GRAPH_HPP
#define READ_GRAPH_HPP

#include <fstream>

#include "init_graph.hpp"

namespace read_graph_from_file {
	uint read_graph(
		std::ifstream& inFile,
		const bool nondirectedGraph,
		const bool firstColumnSourceIndex,	// true if the first column is the source index.
		std::vector<initial_vertex>& initGraph,
		const long long arbparam
		);
}

#endif	//	READ_GRAPH_HPP

#ifndef INIT_GRAPH_HPP
#define INIT_GRAPH_HPP

#include <vector>

#include "../user_specified_codes/user_specified_structures.h"

class neighbor {
public:
	Edge edgeValue;
	unsigned int srcIndex;
};

class initial_vertex {
public:
	Vertex vertexValue;
	Vertex_static VertexValueStatic;
	std::vector<neighbor> nbrs;
	initial_vertex():
		nbrs(0)
	{}
	Vertex& get_vertex_ref() {
		return vertexValue;
	}
};

#endif	//	INIT_GRAPH_HPP

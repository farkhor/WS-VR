#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "read_graph.hpp"
#include "init_graph.hpp"
#include "globals.hpp"
#include "../user_specified_codes/user_specified_pre_and_post_processing_functions.hpp"



uint read_graph_from_file::read_graph(
		std::ifstream& inFile,
		const bool nondirectedGraph,
		const bool firstColumnSourceIndex,	// true if the first column is the source index.
		std::vector<initial_vertex>& initGraph,
		const long long arbparam
		) {

	std::string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;
	uint nEdges = 0;

	unsigned int Additionalargc=0;
	char* Additionalargv[ 61 ];

	// Read the input graph line-by-line.
	while( std::getline( inFile, line ) ) {
		if( line[0] < '0' || line[0] > '9' )	// Skipping any line blank or starting with a character rather than a number.
			continue;
		char cstrLine[256];
		std::strcpy( cstrLine, line.c_str() );
		uint firstIndex, secondIndex;

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			firstIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			secondIndex = atoi( pch );
		else
			continue;

		uint theMax = std::max( firstIndex, secondIndex );
		uint srcVertexIndex = firstColumnSourceIndex ? firstIndex : secondIndex;
		uint dstVertexIndex = firstColumnSourceIndex ? secondIndex : firstIndex;
		if( initGraph.size() <= theMax )
			initGraph.resize(theMax+1);

		Additionalargc=0;
		Additionalargv[ Additionalargc ] = strtok( NULL, delim );
		while( Additionalargv[ Additionalargc ] != NULL ){
			Additionalargc++;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
		}

		{
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			completeEntry(	Additionalargc,
							Additionalargv,
							srcVertexIndex,
							dstVertexIndex,
							&(nbrToAdd.edgeValue),
							(initGraph.at(srcVertexIndex).vertexValue),
							&(initGraph.at(srcVertexIndex).VertexValueStatic),
							(initGraph.at(dstVertexIndex).vertexValue),
							&(initGraph.at(dstVertexIndex).VertexValueStatic),
							arbparam );

			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
		if( nondirectedGraph ) {

			uint tmp = srcVertexIndex;
			srcVertexIndex = dstVertexIndex;
			dstVertexIndex = tmp;

			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			completeEntry(	Additionalargc,
							Additionalargv,
							srcVertexIndex,
							dstVertexIndex,
							&(nbrToAdd.edgeValue),
							initGraph.at(srcVertexIndex).vertexValue,
							&(initGraph.at(srcVertexIndex).VertexValueStatic),
							initGraph.at(dstVertexIndex).vertexValue,
							&(initGraph.at(dstVertexIndex).VertexValueStatic),
							arbparam );

			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}

	}

	uint initialNumVertices = initGraph.size();
	if( ( initialNumVertices % WARP_SIZE ) != 0 )
		initGraph.resize( ( ( initialNumVertices / WARP_SIZE ) + 1 ) * WARP_SIZE );

	return nEdges;

}

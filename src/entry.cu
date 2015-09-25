#include <string>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "utils/init_graph.hpp"
#include "utils/read_graph.hpp"
#include "utils/globals.hpp"
#include "prepare_and_process_graph.cuh"


// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to access specified file: " + file_name + "\n" );
}


int main( int argc, char** argv ) {

	std::string usage =
	"\tRequired command line arguments:\n\
		-Input graph edge list: E.g., --edgelist in.txt\n\
	Additional arguments:\n\
		-Output file (default: out.txt). E.g., --output myout.txt\n\
		-Is the input graph directed (default:yes). To make it undirected: --undirected\n\
		-First column in the edge-list is the indices for source vertices (default: yes). To make the first column standing for the destination vertices --reversedColumns\n\
		-Device ID in single-GPU mode (default: 0). E.g., --device 1\n\
		-Number of GPUs (default: 1). E.g., --nDevices 2.\n\
		-Inter-device communication method between ALL, MS, VRONLINE, and VR(default). E.g., --interdev MS\n\
		-User's arbitrary parameter (default: 0). E.g., --arbparam 17.\n";

	// Required variables for initialization.
	std::ifstream inputEdgeList;
	std::ofstream outputFile;
	bool nonDirectedGraph = false;		// By default, the graph is directed.
	bool firstColumnSourceIndex = true;		// By default, the first column in the graph edge list stand for source vertices.
	long long arbparam = 0;
	int nDevices = 1;
	inter_device_comm commMethod = VR;
	int singleDeviceID = 0;


	/********************************
	 * GETTING INPUT PARAMETERS.
	 ********************************/

	try{

		for( int iii = 1; iii < argc; ++iii )
			if( !strcmp( argv[iii], "--edgelist" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputEdgeList, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp(argv[iii], "--undirected"))
				nonDirectedGraph = true;
			else if( !strcmp(argv[iii], "--reversedColumns"))
				firstColumnSourceIndex = false;
			else if( !strcmp( argv[iii], "--arbparam" ) && iii != argc-1 /*is not the last one*/)
				arbparam = std::atoll( argv[iii+1] );
			else if( !strcmp( argv[iii], "--nDevices" ) && iii != argc-1 /*is not the last one*/)
				nDevices = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--device" ) && iii != argc-1 /*is not the last one*/)
				singleDeviceID = std::atoi( argv[iii+1] );
			else if ( !strcmp(argv[iii], "--interdev") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "VR") )
					commMethod = VR;
				else if ( !strcmp(argv[iii+1], "MS") )
					commMethod = MS;
				else if ( !strcmp(argv[iii+1], "ALL") )
					commMethod = ALL;
				else if ( !strcmp(argv[iii+1], "VRONLINE") )
					commMethod = VRONLINE;
			}

		if( !inputEdgeList.is_open() )
			throw std::runtime_error( "Initialization Error: The input edge list has not been specified." );
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n" << "Usage: " << usage << "\nExiting." << std::endl;
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}


	try {


		if( nDevices > 1 ) {
			std::cout << nDevices << " devices will be processing the graph.\n";
			std::cout << "Inter-device communication scheme will be " <<  ((commMethod==VR)? "VR." : (commMethod==VRONLINE)? "VRONLINE." : (commMethod==MS)? "MS." : "ALL.") << std::endl;
		} else {
			std::cout << "The device with the ID " << singleDeviceID << " will process the graph" << std::endl;
		}

		/********************************
		 * Read the input graph.
		 ********************************/

		std::vector<initial_vertex> inMemGraph(0);
		uint nEdges = read_graph_from_file::read_graph(
				inputEdgeList,
				nonDirectedGraph,		// If the input graph is non-directed, this boolean is true; otherwise it's false.
				firstColumnSourceIndex,		// True if the first column is the source index; otherwise false.
				inMemGraph,		// The read graph.
				arbparam
				);		// Arbitrary user-provided parameter.
		std::cout << "Input graph collected with " << inMemGraph.size() << " vertices and " << nEdges << " edges." << std::endl;


		/********************************
		 * Determine the interval of vertices assigned to each device.
		 ********************************/

		std::vector<unsigned int> indicesRange( nDevices + 1 );
		indicesRange.at(0) = 0;
		indicesRange.at( indicesRange.size() - 1 ) = inMemGraph.size();
		if( nDevices > 1 ){
			uint approxmiateNumEdgesPerDevice = nEdges / nDevices;
			for( unsigned int dev = 1; dev < nDevices; ++dev ) {
				unsigned int accumulatedEdges = 0;
				uint movingVertexIndex = indicesRange.at( dev - 1 );
				while( accumulatedEdges < approxmiateNumEdgesPerDevice ) {
					accumulatedEdges += inMemGraph.at( movingVertexIndex ).nbrs.size();
					++movingVertexIndex;
				}
				movingVertexIndex &= ~( WARP_SIZE - 1 );
				indicesRange.at( dev ) = movingVertexIndex;
			}
		}


		/********************************
		 * Prepare and process the graph.
		 ********************************/

		prepare_and_process_graph(
				&inMemGraph,
				nEdges,
				outputFile,
				&indicesRange,
				nDevices,
				singleDeviceID,
				commMethod );

		std::cout << "Done." << std::endl;
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n" << "Exiting." << std::endl;
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}
}

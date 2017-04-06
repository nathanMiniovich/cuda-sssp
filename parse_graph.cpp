#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <climits>
#include "parse_graph.hpp"

#define SSSP_INF 1073741824

unsigned int count_edges(std::vector<initial_vertex>& graph){

	unsigned int edge_num = 0;

	for(int i = 0 ; i < graph.size() ; i++){
	    edge_num += graph[i].nbrs.size();
	}

	return edge_num;
}

void pull_edges(std::vector<initial_vertex>& graph, edge_node* edge_list, unsigned int edge_num){

	unsigned int k = 0;

	for(int i = 0 ; i < graph.size() ; i++){
	    for(int j = 0 ; j < graph[i].nbrs.size() ; j++, k++){
		edge_list[k].srcIndex = graph[i].nbrs[j].srcIndex;
		edge_list[k].destIndex = i;
		edge_list[k].weight = graph[i].nbrs[j].edgeValue.weight;
	    }
	}

	if( k != edge_num )
	    printf("ERROR: Edge numbers don't match up\n");
}

void pull_distances(unsigned int* dist_arr, int size){

	dist_arr[0] = 0;

	for(int i = 1; i < size; i++){
	    dist_arr[i] = UINT_MAX; 
	}
}

int cmp_edge(const void *a, const void *b){
	
	return ( (int)(((edge_node *)a)->srcIndex) - (int)(((edge_node *)b)->srcIndex));
}

void print_edge_list(edge_node* edge_list, unsigned int edge_num){
	for(int i = 0 ; i < edge_num ; i++){
	    printf("Edge # %d, src: %u, dest: %u, weight: %u\n", i, edge_list[i].srcIndex, edge_list[i].destIndex, edge_list[i].weight);
	}
}

uint parse_graph::parse(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		const long long arbparam,
		const bool nondirected ) {

	const bool firstColumnSourceIndex = true;

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
		{ //This is just a block
		        // Add the neighbor. A neighbor wraps edges
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			initGraph.at(srcVertexIndex).vertexValue.distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).vertexValue.distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.edgeValue.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;

			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
		if( nondirected ) {
		        // Add the edge going the other way
			uint tmp = srcVertexIndex;
			srcVertexIndex = dstVertexIndex;
			dstVertexIndex = tmp;
			//swap src and dest and add as before
			
			neighbor nbrToAdd;
			nbrToAdd.srcIndex = srcVertexIndex;

			Additionalargc=0;
			Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			while( Additionalargv[ Additionalargc ] != NULL ){
				Additionalargc++;
				Additionalargv[ Additionalargc ] = strtok( NULL, delim );
			}
			initGraph.at(srcVertexIndex).vertexValue.distance = ( srcVertexIndex != arbparam ) ? SSSP_INF : 0;
			initGraph.at(dstVertexIndex).vertexValue.distance = ( dstVertexIndex != arbparam ) ? SSSP_INF : 0;
			nbrToAdd.edgeValue.weight = ( Additionalargc > 0 ) ? atoi(Additionalargv[0]) : 1;
			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
	}

	return nEdges;

}

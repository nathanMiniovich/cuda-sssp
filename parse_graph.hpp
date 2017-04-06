#ifndef PARSE_GRAPH_HPP
#define PARSE_GRAPH_HPP

#include <fstream>

#include "initial_graph.hpp"

namespace parse_graph {
	uint parse(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		const long long arbparam,
		const bool nondirected );
}


unsigned int count_edges(std::vector<initial_vertex>& graph);

void pull_edges(std::vector<initial_vertex>& graph, edge_node* edge_list, unsigned int edge_num);

void pull_distances(unsigned int* dist_arr, int size);

int cmp_edge(const void *a, const void *b);

void print_edge_list(edge_node* edge_list, unsigned int edge_num);

#endif	//	PARSE_GRAPH_HPP

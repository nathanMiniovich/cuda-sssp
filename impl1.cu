#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std;

__global__ void edge_process(vector<edge_node>* L, vector<int>* distance_prev, vector<int>* distance_cur, int * anyChange){

}

void swap(vector<int>* distance_cur, vector<int>* distance_prev){

	vector<int> * temp = distance_cur;
	distance_cur = distance_prev;
	distance_prev = temp;
}

vector<edge_node>* pull_edges(vector<initial_vertex>* graph){

	unsigned int edge_num;
	vector<edge_node>* L;

	edge_num = 0;

	for(int i = 0 ; i < graph->size() ; i++){
	    edge_num += graph[i].nbrs.size();
	}

	L = new vector<edge_node>(edge_num);

	for(int i = 0 ; i < graph->size() ; i++){
	    for(int j = 0 ; j < graph[i].nbrs.size() ; j++){
		L[i].srcIndex = i;
		L[i].dstIndex = graph[i].nbrs[j].srcIndex;
		L[i].weight = graph[i].nbrs[j].edgeValue;
	    }
	    
	}

	return L;
}

vector<int>* pull_distances(vector<initial_vertex>* graph){
	
	vector<int>* init_dist = new vector<int>(graph->size());

	init_dist[0] = 0;

	for(int i = 1 ; i < graph->size() ; i++){
	    init_dist[i] = UINT_MAX; 
	}

	return init_dist;
}


void puller(vector<initial_vertex> * graph, int blockSize, int blockNum){
	
	vector<int> initDist, *distance_cur, *distance_prev;
	int *anyChange;
	vector<edge_node> *edge_list, *L;

	initDist = pull_distances(graph);
	edge_list = pull_edges(graph);

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(initDist));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(initDist));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_list));

	cudaMemcpy(distance_cur, &initDist, (size_t)sizeof(initDist), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, &initDist, (size_t)sizeof(initDist), cudaMemcpyHostToDevice);
	cudaMemcpy(L, &edge_list, (size_t)sizeof(edge_list), cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	setTime();

	for(int i=0; i < graph->size()-1; i++){
		pulling_kernel<<<blockNum,blockSize>>>(L, distance_prev, distance_cur, anyChange);
		if(anyChange[0]){
			break;
		} else {
			swap(distance_cur,distance_prev)
		}
	}

	//more housekeeping needed

	cout << "Took " << getTime() << "ms.\n";

	// write to output.txt 
}

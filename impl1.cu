#include <vector>
#include <iostream>
#include <climits>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include <algorithm>

using namespace std;

__global__ void edge_process(const edge_node *L, const unsigned int edge_num, unsigned int *distance_prev, unsigned int *distance_cur, int* anyChange){
	
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;

	int warp_id = thread_id % 32 ? thread_id/32 + 1: thread_id/32;
	int warp_num = thread_num % 32 ? thread_num/32 + 1: thread_num/32;
	int lane_id = thread_id % 32;

	int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
	int beg = load % warp_id;
	int end = min(edge_num, beg + load);
	beg += lane_id;

	unsigned int u;
	unsigned int v;
	unsigned int w;

	for(int i = beg; i < end; i+=32){
		u = L[i].srcIndex;
		v = L[i].destIndex;
		w = L[i].weight;
		if(distance_prev[u] != UINT_MAX){
		    if((distance_prev[u] + w) < distance_prev[v]){
			if(distance_prev[u] + w < distance_cur[v]){
			    anyChange[0] = 1;
			}
			atomicMin(&distance_cur[v], distance_prev[u] + w);
		    }
		}
	}
}


unsigned int count_edges(vector<initial_vertex>& graph){

	unsigned int edge_num = 0;

	for(int i = 0 ; i < graph.size() ; i++){
	    edge_num += graph[i].nbrs.size();
	}

	return edge_num;
}

void pull_edges(vector<initial_vertex>& graph, edge_node* edge_list, unsigned int edge_num){

	unsigned int k = 0;

	for(int i = 0 ; i < graph.size() ; i++){
	    for(int j = 0 ; j < graph[i].nbrs.size() ; j++, k++){
		edge_list[k].srcIndex = i;
		edge_list[k].destIndex = graph[i].nbrs[j].srcIndex;
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

void puller(vector<initial_vertex> * graph, int blockSize, int blockNum){

	unsigned int *initDist, *distance_cur, *distance_prev; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	edge_node *edge_list, *L;
	unsigned int edge_num;
	
	edge_num = count_edges(*graph);
	edge_list = (edge_node*) malloc(sizeof(edge_node)*edge_num);
	initDist = (unsigned int*)calloc(graph->size(),sizeof(unsigned int));	
	pull_distances(initDist, graph->size());
	pull_edges(*graph, edge_list, edge_num);

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_node)*edge_num);

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(edge_node)*edge_num, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	setTime();

	for(int i=0; i < ((int) graph->size())-1; i++){
		edge_process<<<blockNum,blockSize>>>(L, edge_num, distance_prev, distance_cur, anyChange);
		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!hostAnyChange[0]){
			break;
		} else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
			memcpy(distance_prev, distance_cur,(size_t)graph->size());
		}
	}

	//more housekeeping needed
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	cudaFree(L);
	
	free(initDist);
	free(edge_list);

	cout << "Took " << getTime() << "ms.\n";

	// write to output.txt 
}

#include <vector>
#include <iostream>
#include <climits>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include <algorithm>

using namespace std;

__global__ void edge_process(vector<edge_node>* L, vector<int>* distance_prev, vector<int>* distance_cur, int* anyChange){

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;

	int warp_id = thread_id % 32 ? thread_id/32 + 1: thread_id/32;
	int warp_num = thread_num % 32 ? thread_num/32 + 1: thread_num/32;
	int lane_id = thread_id % 32;

	int load = (L->size() % warp_num == 0) ? L->size()/warp_num : L->size()/warp_num+1;
	int beg = load % warp_id;
	int end = min(L->size(), beg + load);
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
			    *anyChange = 1;
			}
			atomicMin(&distance_cur[v], distance_prev[u] + w);
		    }
		}
	}
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
	
	vector<unsigned int> initDist, *distance_cur, *distance_prev;
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
		if(!anyChange[0]){
			break;
		} else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
			distance_curr->swap(*distance_prev)
		}
	}

	//more housekeeping needed

	cout << "Took " << getTime() << "ms.\n";

	// write to output.txt 
}

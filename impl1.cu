#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include <algorithm>

using namespace std;

__global__ void edge_process(vector<edge_node>* L, vector<int>* distance_prev, vector<int>* distance_cur, int* anyChange){

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;

	int warp_id = thread_id/32;
	int warp_num = thread_num/32;
	int lane_id = thread_id % 32;

	load = (L.size() % warp_num == 0) ? L.size()/warp_num : L.size()/warp_num+1;
	beg = load % warp_id;
	end = min(L.size(), beg + load);
	beg += lane_id;

	unsigned int u;
	unsigned int v;
	unsigned int w;

	for(i = beg; i < end; i+=32){
		u = L[i].srcIndex;
		v = L[i].destIndex;
		w = L[i].weight;
		
		if((distance_prev[u] + w) < distance_prev[v]){
			atomicMin(&distance_cur[v], distance_prev[u] + w);
		}
	}
}

void swap(vector<int>* distance_cur, vector<int>* distance_prev){

	vector<int> * temp = distance_cur;
	distance_cur = distance_prev;
	distance_prev = temp;
}

void puller(vector<initial_vertex> * graph, int blockSize, int blockNum){
	
	vector<int> hostInitDist(graph->size());
	vector<int> * distance_cur, * distance_prev;
	int * anyChange;

	for(int i=0; i < graph->size(); i++){
	    hostInitDist[i] = graph[i].vertexValue.distance;
	}

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(hostInitDist));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(hostInitDist));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));

	cudaMemcpy(distance_cur, &hostInitDist, (size_t)sizeof(hostInitDist), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, &hostInitDist, (size_t)sizeof(hostInitDist), cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	setTime();

	for(int i=0; i < graph->size()-1; i++){
		pulling_kernel<<<blockNum,blockSize>>>(graph, distance_prev, distance_cur, 0, anyChange);
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

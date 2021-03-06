#include <vector>
#include <iostream>
#include <climits>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include <algorithm>

#define INF 1073741824
#define MAX_PER_BLOCK 1024 
#define FILL UINT_MAX - 777

using namespace std;

// dests[i]: destination index of edge_list[i] (block size)
// vals[i]: temporary distance TO dests[i]     (block size)
// distance_cur: good 'ol distance array (size |V|)
__device__ void segmented_scan_min(const int lane, const unsigned int *dests, unsigned int *vals, unsigned int *distance_cur){

    if ( lane >= 1 && dests[threadIdx.x] == dests[threadIdx.x - 1] )
	vals[threadIdx.x] = min(vals[threadIdx.x], vals[threadIdx.x - 1]);

    if ( lane >= 2 && dests[threadIdx.x] == dests[threadIdx.x - 2] )
	vals[threadIdx.x] = min(vals[threadIdx.x], vals[threadIdx.x - 2]);

    if ( lane >= 4 && dests[threadIdx.x] == dests[threadIdx.x - 4] )
	vals[threadIdx.x] = min(vals[threadIdx.x], vals[threadIdx.x - 4]);

    if ( lane >= 8 && dests[threadIdx.x] == dests[threadIdx.x - 8] )
	vals[threadIdx.x] = min(vals[threadIdx.x], vals[threadIdx.x - 8]);

    if ( lane >= 16 && dests[threadIdx.x] == dests[threadIdx.x - 16] )
	vals[threadIdx.x] = min(vals[threadIdx.x], vals[threadIdx.x - 16]);

    if ( lane == 31 || dests[threadIdx.x] != dests[threadIdx.x + 1] || dests[threadIdx.x + 1] == FILL )
	atomicMin(&distance_cur[dests[threadIdx.x]], vals[threadIdx.x]);
}

__global__ void edge_process_usesmem(const edge_node *L, const unsigned int edge_num, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange){
	__shared__ unsigned int dests[MAX_PER_BLOCK];
	__shared__ unsigned int vals[MAX_PER_BLOCK];

	dests[threadIdx.x] = FILL;

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;

	int warp_id = thread_id/32;
	int warp_num = thread_num % 32 ? thread_num/32 + 1 : thread_num/32;
	int lane_id = thread_id % 32;

	int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
	int beg = load * warp_id;
	int end = min(edge_num, beg + load);
	beg += lane_id;

	unsigned int u;
	unsigned int v;
	unsigned int w;
	unsigned int temp;

	for(int i = beg; i < end; i+=32){
		u = L[i].srcIndex;
		v = L[i].destIndex;
		w = L[i].weight;

		dests[threadIdx.x] = v;
		temp = distance_cur[v];
		
		if(distance_prev[u] == UINT_MAX){
		    vals[threadIdx.x] = UINT_MAX;
		} else {
		    vals[threadIdx.x] = distance_prev[u] + w;
		}

		segmented_scan_min(thread_id % 32, dests, vals, distance_cur);

		if(distance_cur[v] < temp)
		    anyChange[0] = 1;

		dests[threadIdx.x] = FILL ;
	}
}

__global__ void edge_process_incore(const edge_node *L, const unsigned int edge_num, unsigned int *distance, int *anyChange){

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;

	int warp_id = thread_id/32;
	int warp_num = thread_num % 32 ? thread_num/32 + 1 : thread_num/32;
	int lane_id = thread_id % 32;

	int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
	int beg = load * warp_id;
	int end = min(edge_num, beg + load);
	beg += lane_id;

	unsigned int u;
	unsigned int v;
	unsigned int w;

	for(int i = beg; i < end; i+=32){
		u = L[i].srcIndex;
		v = L[i].destIndex;
		w = L[i].weight;
		int dist = distance[u]+w;
		if(distance[u] == UINT_MAX){
			continue;
		} else if(dist < distance[v]){
			anyChange[0] = 1;
			atomicMin(&distance[v], dist);
		}
	}
}

__global__ void edge_process(const edge_node *L, const unsigned int edge_num, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange){
	
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;

	int warp_id = thread_id/32;
	int warp_num = thread_num % 32 ? thread_num/32 + 1 : thread_num/32;
	int lane_id = thread_id % 32;

	int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
	int beg = load * warp_id;
	int end = min(edge_num, beg + load);
	beg += lane_id;

	unsigned int u;
	unsigned int v;
	unsigned int w;

	for(int i = beg; i < end; i+=32){
		u = L[i].srcIndex;
		v = L[i].destIndex;
		w = L[i].weight;
		if(distance_prev[u] == UINT_MAX){
			continue;
		} else if(distance_prev[u] + w < distance_cur[v]){
			anyChange[0] = 1;
			atomicMin(&distance_cur[v], distance_prev[u] + w);
		}
	}
}

void puller_usesmem(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile){

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

	//print_edge_list(edge_list, edge_num);

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_node)*edge_num);

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(edge_node)*edge_num, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	setTime();

	for(int i=0; i < ((int) graph->size()) - 1; i++){
		edge_process_usesmem<<<blockNum,blockSize>>>(L, edge_num, distance_prev, distance_cur, anyChange);
		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!hostAnyChange[0]){
			break;
		} else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
			cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
			cudaMemcpy(distance_cur, distance_prev, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
			cudaMemcpy(distance_prev, hostDistanceCur,(sizeof(unsigned int))*(graph->size()), cudaMemcpyHostToDevice);
		}
	}

	cout << "Took " << getTime() << "ms.\n";

	cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(hostDistanceCur[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}else{
		    outputFile << i << ":" << hostDistanceCur[i] << endl; 
		}
	}

	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	cudaFree(L);
	
	delete[] hostDistanceCur;
	free(initDist);
	free(edge_list);
}

void puller(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile, bool sortBySource){

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

	if(sortBySource){
	    qsort(edge_list, edge_num, sizeof(edge_node), cmp_edge);
	}

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

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
			cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
			cudaMemcpy(distance_cur, distance_prev, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
			cudaMemcpy(distance_prev, hostDistanceCur,(sizeof(unsigned int))*(graph->size()), cudaMemcpyHostToDevice);
		}
	}

	cout << "Took " << getTime() << "ms.\n";

	cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(hostDistanceCur[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}else{
		    outputFile << i << ":" << hostDistanceCur[i] << endl; 
		}
	}

	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	cudaFree(L);
	
	delete[] hostDistanceCur;
	free(initDist);
	free(edge_list);
}

void puller_incore(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile, bool sortBySource){

	unsigned int *initDist, *distance; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	edge_node *edge_list, *L;
	unsigned int edge_num;
	
	edge_num = count_edges(*graph);
	edge_list = (edge_node*) malloc(sizeof(edge_node)*edge_num);
	initDist = (unsigned int*)calloc(graph->size(),sizeof(unsigned int));	
	pull_distances(initDist, graph->size());
	pull_edges(*graph, edge_list, edge_num);

	if(sortBySource){
	    qsort(edge_list, edge_num, sizeof(edge_node), cmp_edge);
	}

	cudaMalloc((void**)&distance, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_node)*edge_num);

	cudaMemcpy(distance, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(edge_node)*edge_num, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	setTime();

	for(int i=0; i < ((int) graph->size())-1; i++){
		edge_process_incore<<<blockNum,blockSize>>>(L, edge_num, distance, anyChange);
		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!hostAnyChange[0]){
			break;
		} else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
		}
	}

	cout << "Took " << getTime() << "ms.\n";

	unsigned int *hostDistance = (unsigned int *)malloc((sizeof(unsigned int))*(graph->size()));	
	cudaMemcpy(hostDistance, distance, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(hostDistance[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}else{
		    outputFile << i << ":" << hostDistance[i] << endl; 
		}
	}

	cudaFree(distance);
	cudaFree(anyChange);
	cudaFree(L);
	
	delete[] hostDistance;
	free(initDist);
	free(edge_list);
}

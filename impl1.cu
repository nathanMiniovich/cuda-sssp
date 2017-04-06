#include <vector>
#include <iostream>
#include <climits>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include <algorithm>

#define INF 1073741824

using namespace std;

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

void puller(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile){

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

	unsigned int *hostDistanceCurr = (unsigned int *)malloc((sizeof(unsigned int))*(graph->size()));	
	cudaMemcpy(hostDistanceCurr, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		outputFile << i << ":" << hostDistanceCurr[i] << endl; 
	}

	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	cudaFree(L);
	
	delete[] hostDistanceCur;
	free(initDist);
	free(edge_list);
}

void puller_incore(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile){

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

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

	cudaMalloc((void**)&distance, (size_t)sizeof(unsigned int)*(graph->size()));
	//cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_node)*edge_num);

	cudaMemcpy(distance, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	//cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
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
			//cudaMemcpy(hostDistance, distance, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
			//cudaMemcpy(distance_cur, distance_prev, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
			//cudaMemcpy(distance_prev, hostDistanceCur,(sizeof(unsigned int))*(graph->size()), cudaMemcpyHostToDevice);
		}
	}

	cout << "Took " << getTime() << "ms.\n";

	unsigned int *hostDistance = (unsigned int *)malloc((sizeof(unsigned int))*(graph->size()));	
	cudaMemcpy(hostDistance, distance, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		outputFile << i << ":" << hostDistance[i] << endl; 
	}

	cudaFree(distance);
	//cudaFree(distance_prev);
	cudaFree(anyChange);
	cudaFree(L);
	
	delete[] hostDistance;
	free(initDist);
	free(edge_list);
}

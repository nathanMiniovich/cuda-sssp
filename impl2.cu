#include <vector>
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <thrust/scan.h>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std;

// OUTCORE
__global__ void edge_process_outcore(const edge_node *L, const unsigned int edge_num, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange, unsigned int *pred){
	
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
			pred[v] = 1;
			atomicMin(&distance_cur[v], distance_prev[u] + w);
		}
	}
}

// INCORE 
__global__ void edge_process_incore(const edge_node *L, const unsigned int edge_num, unsigned int *distance, int *anyChange, unsigned int *pred){

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
		int dist = distance[u] + w;
		if(distance[u] == UINT_MAX){
			continue;
		} else if(dist < distance[v]){
			//printf("src is %u , dest is %u, weight is %u\n", u, v, w);

			anyChange[0] = 1;
			pred[v] = 1;
			atomicMin(&distance[v], dist);
		}
	}
}

__global__ void getX(const edge_node *L, const unsigned int edge_num, unsigned int *pred, unsigned int *X){
    
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;

    int warp_id = thread_id/32;
    int warp_num = (thread_num % 32 == 0) ? thread_num/32 : edge_num/32 + 1;
    int lane_id = thread_id % 32;

    // how many edges each warp takes
    int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
    int beg = load * warp_id;
    int end = min(edge_num, beg+ load);
    beg += lane_id;

    unsigned int num = 0;
    for(int i = beg; i < end; i+=32){
	int mask = __ballot(pred[L[i].srcIndex]);
	if(lane_id == 0){
	    num += (unsigned int) __popc(mask);
	}
    }

    if(lane_id == 0){
	X[warp_id] = num;
    }
}

__global__ void getY(unsigned int *X){
   
    int n = blockDim.x;
    int thid = threadIdx.x;
    int offset = 1;

    for(int d = n >> 1; d > 0; d >>= 1){
	__syncthreads();
	if( thid < d ){
	    int ai = offset*(2*thid+1)-1;
	    int bi = offset*(2*thid+2)-1;
	    X[bi] += X[ai];
	}
	offset *= 2;
    }
    
    if( thid == 0) { X[n-1] = 0; }

    for(int d = 1 ; d < n ; d *= 2){
	offset >>= 1;
	__syncthreads();
	if( thid < d ){
	    int ai = offset*(2*thid+1)-1;
	    int bi = offset*(2*thid+2)-1;
	    int t = X[ai];
	    X[ai] = X[bi];
	    X[bi] += t;
	}
    }
    __syncthreads();
}

__global__ void getT(const edge_node *L, const unsigned int edge_num, unsigned int *pred, unsigned int *Y, edge_node *T){
    // fill here
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;

    int warp_id = thread_id/32;
    int warp_num = (thread_num % 32 == 0) ? thread_num/32 : edge_num/32 + 1;
    int lane_id = thread_id % 32;

    // how many edges each warp takes
    int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
    int beg = load * warp_id;
    int end = min(edge_num, beg+ load);
    beg += lane_id;
    int cur_offset = Y[warp_id];
    
    for(int i = beg; i < end; i+=32){
	int mask = __ballot(pred[L[i].srcIndex]);
	int local_id = __popc(mask << (32 - 1) - lane_id) - 1;
	if(pred[L[i].srcIndex]){
	    T[cur_offset+local_id]= L[i];
	}
	cur_offset += __popc(mask);
    }

}

// OUTCORE
void impl2_outcore(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile, bool sortBySource){

	double t_filter, t_comp;
	t_filter = 0;
	t_comp = 0;

	unsigned int *initDist, *distance_cur, *distance_prev, *to_process_arr, *pred; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	edge_node *edge_list, *L, *T;
	unsigned int edge_num, to_process_num;
	unsigned int *temp = (unsigned int*)malloc(sizeof(unsigned int));
	
	int thread_num = blockSize * blockNum;
	edge_num = count_edges(*graph);
	int warp_num = (thread_num % 32 == 0) ? thread_num/32 : thread_num/32 + 1;
	edge_list = (edge_node*) malloc(sizeof(edge_node)*edge_num);
	initDist = (unsigned int*)calloc(graph->size(),sizeof(unsigned int));	
	pull_distances(initDist, graph->size());
	pull_edges(*graph, edge_list, edge_num);

	if(sortBySource){
	    qsort(edge_list, edge_num, sizeof(edge_node), cmp_edge);
	}

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];
	unsigned int *hostTPA = new unsigned int[warp_num];

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_node)*edge_num);
	cudaMalloc((void**)&to_process_arr, (size_t)sizeof(unsigned int)*warp_num);
	cudaMalloc((void**)&pred, (size_t)sizeof(unsigned int)*(graph->size()));

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(edge_node)*edge_num, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));
	cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);
	cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));

	for(int i=0; i < ((int) graph->size())-1; i++){

		setTime();

		if(i == 0){
		    edge_process_outcore<<<blockNum,blockSize>>>(L, edge_num, distance_prev, distance_cur, anyChange, pred);
		    cudaDeviceSynchronize();
		} else {
		    cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));
		    edge_process_outcore<<<blockNum,blockSize>>>(T, to_process_num, distance_prev, distance_cur, anyChange, pred);
		    cudaDeviceSynchronize();
		    cudaFree(T);
		}

		t_comp += getTime();

		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);

		if(!hostAnyChange[0]){
			break;
		} else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
			cudaMemcpy(distance_prev, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
			cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
		}

		if(i == graph->size() - 2){
		    break;
		} else {

		    setTime();

		    cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);

		    getX<<<blockNum, blockSize>>>(L, edge_num, pred, to_process_arr);
		    cudaDeviceSynchronize();

		    cudaMemcpy(temp, to_process_arr + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		    to_process_num = *temp;

		    cudaMemcpy(hostTPA, to_process_arr, sizeof(unsigned int)*warp_num, cudaMemcpyDeviceToHost);
		    thrust::exclusive_scan(hostTPA, hostTPA + warp_num, hostTPA);
		    cudaMemcpy(to_process_arr, hostTPA, sizeof(unsigned int)*warp_num, cudaMemcpyHostToDevice);

		    cudaMemcpy(temp, to_process_arr + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num += *temp;

		    cudaMalloc((void**)&T, (size_t)sizeof(edge_node)*to_process_num);
		    
		    getT<<<blockNum, blockSize>>>(L, edge_num, pred, to_process_arr, T);
		    cudaDeviceSynchronize();

		    t_filter += getTime();
		}
	}

	printf("Computation Time: %f ms\nFiltering Time: %f ms\n", t_comp, t_filter);

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
	
	delete[] hostTPA;
	delete[] hostDistanceCur;
	free(initDist);
	free(edge_list);
}


// INCORE
void impl2_incore(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile, bool sortBySource){

	double t_filter, t_comp;
	t_comp = 0;
	t_filter = 0;

	unsigned int *initDist, *distance, *to_process_arr, *pred; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	edge_node *edge_list, *L, *T;
	unsigned int edge_num, to_process_num;
	unsigned int *temp = (unsigned int*)malloc(sizeof(unsigned int));

	int thread_num = blockSize * blockNum;
	edge_num = count_edges(*graph);
	int warp_num = (thread_num % 32 == 0) ? thread_num/32 : thread_num/32 + 1;
	edge_list = (edge_node*) malloc(sizeof(edge_node)*edge_num);
	initDist = (unsigned int*)calloc(graph->size(),sizeof(unsigned int));	
	pull_distances(initDist, graph->size());
	pull_edges(*graph, edge_list, edge_num);

	unsigned int *hostTPA = new unsigned int[warp_num];

	if(sortBySource){
	    qsort(edge_list, edge_num, sizeof(edge_node), cmp_edge);
	}

	cudaMalloc((void**)&distance, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(edge_node)*edge_num);
	cudaMalloc((void**)&to_process_arr, (size_t)sizeof(unsigned int)*warp_num);
	cudaMalloc((void**)&pred, (size_t)sizeof(unsigned int)*(graph->size()));

	cudaMemcpy(distance, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(edge_node)*edge_num, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));
	cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);
	cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));

	for(int i=0; i < ((int) graph->size())-1; i++){
		
		setTime();

		if( i == 0 ){
		    edge_process_incore<<<blockNum,blockSize>>>(L, edge_num, distance, anyChange, pred);
		} else {
		    cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));
		    edge_process_incore<<<blockNum, blockSize>>>(T, to_process_num, distance, anyChange, pred);
		    cudaFree(T);

		}

		t_comp += getTime();

		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!hostAnyChange[0]){
		    break;
		} else {
		    cudaMemset(anyChange, 0, (size_t)sizeof(int));
		}

		if(i == graph->size() - 2){
		    break;
		} else {

		    setTime();

		    cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);

		    getX<<<blockNum, blockSize>>>(L, edge_num, pred, to_process_arr);
		    cudaDeviceSynchronize();

		    cudaMemcpy(temp, to_process_arr + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		    to_process_num = *temp;

		    cudaMemcpy(hostTPA, to_process_arr, sizeof(unsigned int)*warp_num, cudaMemcpyDeviceToHost);
		    thrust::exclusive_scan(hostTPA, hostTPA + warp_num, hostTPA);
		    cudaMemcpy(to_process_arr, hostTPA, sizeof(unsigned int)*warp_num, cudaMemcpyHostToDevice);
		    cudaMemcpy(temp, to_process_arr + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		    to_process_num += *temp;

		    cudaMalloc((void**)&T, (size_t)sizeof(edge_node)*to_process_num);

		    getT<<<blockNum, blockSize>>>(L, edge_num, pred, to_process_arr, T);
		    cudaDeviceSynchronize();

		    t_filter += getTime();
		}
	}

	printf("Computation Time: %f ms\nFiltering Time: %f ms\n", t_comp, t_filter);

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
	cudaFree(to_process_arr);
		        
	delete[] hostTPA;
	delete[] hostDistance;
	free(initDist);
	free(edge_list);
}


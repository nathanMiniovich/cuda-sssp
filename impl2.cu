#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

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
			anyChange[0] = 1;
			pred[L[i].srcIndex] = 1
			atomicMin(&distance[v], dist);
		}
	}
}
__global__ void neighborHandling_kernel(std::vector<initial_vertex> * peeps, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //Enqueue and dequeue me as needed.
    //Offset will tell you who I am.
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
    int loop_iter = 1;
    
    for(int i = beg; i < end; i+=32){
	int mask = __ballot(pred[L[i].srcIndex]);
	int local_id = __popc(mask << (32 - 1) - lane_id) - 1;
	T[cur_offset+local_id]= Y[i];
	curr_offset += __popc(mask);
    }

}

void to_process_incore(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream& outputFile){

	unsigned int *initDist, *distance, *to_process_arr, *pred; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	edge_node *edge_list, *L, *T;
	unsigned int edge_num, to_process_num;
	int warp_num = (thread_num % 32 == 0) ? thread_num/32 : edge_num/32 + 1;
	
	edge_num = count_edges(*graph);
	edge_list = (edge_node*) malloc(sizeof(edge_node)*edge_num);
	initDist = (unsigned int*)calloc(graph->size(),sizeof(unsigned int));	
	pull_distances(initDist, graph->size());
	pull_edges(*graph, edge_list, edge_num);

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

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

	setTime();

	for(int i=0; i < ((int) graph->size())-1; i++){
		// begin compute
		if( i == 0 ){
		    edge_process_incore<<<blockNum,blockSize>>>(L, edge_num, distance, anyChange, pred);
		} else {
		    cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));
		    edge_process_incore<<<blockNum, blockSize>>>(T, edge_num, distance, anyChange, pred);
		    cudaFree(T);
		}
		// end compute
		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!hostAnyChange[0]){
		    break;
		} else {
		    cudaMemset(anyChange, 0, (size_t)sizeof(int));
		}

		if(i == graph->size() - 2){
		    break;
		} else {
		    // begin filter
		    cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);

		    getX<<<blockNum, blockSize>>>(L, edge_num, pred, to_process_arr);
		    to_process_num = to_process_arr[warp_num - 1];
		    getY<<<1, warp_num>>>(to_process_arr);
		    to_process_num += to_process[warp_num - 1];

		    cudaMalloc((void**)&T, (size_t)sizeof(edge_node)*to_process_num);

		    getT(L, edge_num, pred, to_process_arr, T);
		    // end filter
		}
	}

	cout << "Took " << getTime() << "ms.\n";

	unsigned int *hostDistance = (unsigned int *)malloc((sizeof(unsigned int))*(graph->size()));	
	cudaMemcpy(hostDistance, distance, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		outputFile << i << ":" << hostDistance[i] << endl; 
	}

	cudaFree(distance);
	cudaFree(anyChange);
	cudaFree(L);
	cudaFree(to_process_arr);
		        
	delete[] hostDistance;
	free(initDist);
	free(edge_list);
}

/*void neighborHandler(std::vector<initial_vertex> * peeps, int blockSize, int blockNum){


    int warp_num = (thread_num % 32 == 0) ? thread_num/32 : edge_num/32 + 1;
    
    setTime();
    getX<<blockNum, blockSize>>(L, edge_num, pred, to_process);
    pseudo code
    int num_to_process = to_process[warp_num - 1];
    
    getY<<1, warp_num>>(to_process);
    
    num_to_process += to_process[warp_num - 1];
    create T
    
    std::cout << "Filtering Stage Took " << getTime() << "ms.\n";


}*/

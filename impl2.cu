#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

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
}

void neighborHandler(std::vector<initial_vertex> * peeps, int blockSize, int blockNum){

    /*
     * Do all the things here!
     *
     */

    int warp_num = (thread_num % 32 == 0) ? thread_num/32 : edge_num/32 + 1;
    
    setTime();
    getX<<blockNum, blockSize>>(L, edge_num, pred, to_process);
    getY<<1, warp_num>>(to_process);
    std::cout << "Filtering Stage Took " << getTime() << "ms.\n";


}

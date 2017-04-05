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

        
// step 1
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

__global__ void getY(unsigned int *X, unsigned int *Y){

}

void neighborHandler(std::vector<initial_vertex> * peeps, int blockSize, int blockNum){
    setTime();

    /*
     * Do all the things here!
     *
     */



    std::cout << "Took " << getTime() << "ms.\n";
}

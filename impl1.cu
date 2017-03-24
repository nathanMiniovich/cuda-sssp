#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std

__global__ void edge_process(vector<initial_vertex> * neighbours, vector<int>* distance_prev, vector<int>* distance_cur, int offset, int * anyChange){

	//update me based on my neighbors. Toggle anyChange as needed.
	//offset will tell you who I am.
}

void swap(vector<int>* distance_cur, vector<int>* distance_prev){
	
}

void puller(vector<initial_vertex> * neighbours, int blockSize, int blockNum){
	setTime();

	//housekeeping goes here
	//populate distance_prev & distance_cur
	
	for(int i=0; i < peeps->size()-1; i++){
		pulling_kernel<<<blockNum,blockSize>>>(neighbours, distance_prev, distance_cur, 0, anyChange);
		if(anyChange[0]){
			break;
		} else {
			swap(distance_cur,distance_prev)
		}
	}

	//more housekeeping needed

	cout << "Took " << getTime() << "ms.\n";
}

#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std;

__global__ void edge_process(vector<initial_vertex> * graph, vector<int>* distance_prev, vector<int>* distance_cur, int offset, int * anyChange){

	//update me based on my neighbors. Toggle anyChange as needed.
	//offset will tell you who I am.
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

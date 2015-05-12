
#ifndef NSSP_H_
#define NSSP_H_
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>

void CudaMallocErrorCheck(void** ptr, int size);
void DijkstrasSetupCuda(int *V, int *E, int *We, int *sigma, int *F, int *U, int num_v, int num_e);
void Extremas(int *V, int *E, int num_v, int num_e, int *extrema_vertex, int source_vertex);
void Initialize(int *V, int *E, int num_v, int num_e, int **dev_V, int **dev_E, int **dev_U, int **dev_F, int **dev_sigma, int source);
int Minimum(int *U, int *sigma, int *V, int *E, int num_v, int num_e, int *dev_dest, int *dev_src);

#endif
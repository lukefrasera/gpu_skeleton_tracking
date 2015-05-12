#include "gpu-nsssp.h"

__global__ void InitializeGPU(int *V, int *E, int *U, int *F, int *sigma, int src, int size_v, int size_e);
__global__ void Relax(int *U, int *F, int *sigma, int *V, int *E, int num_v, int num_e);
__global__ void Update(int *U, int *F, int *sigma, int delta, int size);

__global__ void reduce(int *g_idata, int *g_odata, unsigned int n, int *U, int *sigma);
__global__ void reduce_fix(int *g_idata, int *g_odata, unsigned int n, unsigned int s_size, unsigned int loops, int *U, int *sigma);
uint32_t NearestPowerTwo(uint32_t N);
uint32_t NearestPowerBase(uint32_t N, uint32_t base, uint32_t &power);

void DijkstrasSetupCuda(int *V, int *E, int *We, int *sigma, int *F, int *U, int num_v, int num_e) {
  int extrema_vertex;
  Extremas(V, E, num_v, num_e, &extrema_vertex, 0);
}

void Extremas(int *V, int *E, int num_v, int num_e, int *extrema_vertex, int source_vertex) {
  // Define Unsettled sigma and Frontier nodes
  int *dev_U, *dev_sigma, *dev_F, *dev_V, *dev_E, *dev_src, *dev_dest;
  int delta = 0;
  float elapsedTime=0;
  int size = 80;
  int delta_hist[size];
  int delta_pos = 0;
  // for (int i = 0; i < num_v-1; ++i) {
  //   printf("V[%d]: Edges\n", i);
  //   for (int j = V[i]; j<V[i+1]; ++j) {
  //     printf(", %d", E[j]);
  //   }
  //   printf("\n");
  // }

  // Initialize reduce function mem
  CudaMallocErrorCheck((void**)&dev_src, num_v*sizeof(int));
  CudaMallocErrorCheck((void**)&dev_dest, num_v*sizeof(int));

  Initialize(V, E, num_v, num_e, &dev_V, &dev_E, &dev_U, &dev_F, &dev_sigma, source_vertex);

  // printf("Initialized\n");
//  Relax<<<1, 5>>>(dev_U, dev_F, dev_sigma, dev_V, dev_E, num_v, num_e);
//  int test = Minimum(dev_U, dev_sigma, dev_V, dev_E, num_v, num_e, dev_dest, dev_src);
//  Update<<<1,5>>>(dev_U, dev_F, dev_sigma, test, num_v);
//  printf("Test: %d\n", test);
//
  // printf("C: %d, NumEdges:%d\n", source_vertex, V[source_vertex+1] - V[source_vertex]);
  // for (int i = 0; i < V[source_vertex+1] - V[source_vertex]; ++i) {
  //   printf("V_n = %d ", E[V[i+source_vertex]]);
  // }
  // printf("\n");
//  int V_t[num_v];
//  int U_t[num_v];
//  cudaMemcpy(V_t, dev_F, num_v*sizeof(int), cudaMemcpyDeviceToHost);
//  cudaMemcpy(U_t, dev_U, num_v*sizeof(int), cudaMemcpyDeviceToHost);
  // int counter = 0;
  // cudaEvent_t start, end;
  // cudaEventCreate(&start);
  // cudaEventCreate(&end);
  // cudaEventRecord(start, 0);
  int sigma[num_v];
  while (delta != INT_MAX) { 
    Relax<<<1, num_v>>>(dev_U, dev_F, dev_sigma, dev_V, dev_E, num_v, num_e);
    delta = Minimum(dev_U, dev_sigma, dev_V, dev_E, num_v, num_e, dev_dest, dev_src);
    delta_hist[delta_pos] = delta;
    delta_pos = (delta_pos+1)%size;
    uint32_t sum=0;
    for (int i = 0; i < size; ++i) {
      sum += delta_hist[i];
      // printf("Delta Hist: %d ", delta_hist[i]);
    }
    // printf("\n");
    sum /=size;
    if (delta_hist[0] == sum) {
      break;
    }
    Update<<<1, num_v>>>(dev_U, dev_F, dev_sigma, delta, num_v);
  }
  // cudaEventRecord(end, 0);
  // cudaEventSynchronize(end);
  // cudaEventElapsedTime(&elapsedTime, start, end);
  // printf("Elapsed Time: %f\n", elapsedTime);
  // cudaEventDestroy(start);
  // cudaEventDestroy(end);

}

void Initialize(int *V, int *E, int num_v, int num_e, int **dev_V, int **dev_E, int **dev_U, int **dev_F, int **dev_sigma, int source) {
  // Allocate the device memory
  CudaMallocErrorCheck((void**)dev_V, num_v*sizeof(int));
  CudaMallocErrorCheck((void**)dev_E, num_e*sizeof(int));
  CudaMallocErrorCheck((void**)dev_U, num_v*sizeof(int));
  CudaMallocErrorCheck((void**)dev_F, num_v*sizeof(int));
  CudaMallocErrorCheck((void**)dev_sigma, num_v*sizeof(int));

  // copy graph to device
  cudaMemcpy(*dev_V, V, num_v*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(*dev_E, E, num_e*sizeof(int), cudaMemcpyHostToDevice);
  // initialize Frontier
  // Initialize Unselttled
  // Initialize Sigma distance function
  int threads_per_block, blocks_per_dim;
  blocks_per_dim = num_v / 1024 + 1;
  threads_per_block = num_v / blocks_per_dim;

  InitializeGPU<<<blocks_per_dim, threads_per_block>>>(*dev_V, *dev_E, *dev_U, *dev_F, *dev_sigma, source, num_e, num_v);
}

__global__ void InitializeGPU(int *V, int *E, int *U, int *F, int *sigma, int src, int size_v, int size_e) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int U_t, F_t, sigma_t;

  if (offset < size_v) {
    U_t = 1;
    F_t = 0;
    sigma_t = INT_MAX;

    if (offset == src) {
      U_t = 0;
      F_t = 1;
      sigma_t = 0;
    }
    U[offset] = U_t;
    F[offset] = F_t;
    sigma[offset] = sigma_t;
  }
}

__global__ void Relax(int *U, int *F, int *sigma, int *V, int *E, int num_v, int num_e) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;

  if (offset < num_v) {
    if (F[offset] == 1) {
      for (int i = V[offset]; i < V[offset+1] && i < num_e; ++i) {
        if (U[E[i]] == 1) {
          atomicMin(&sigma[E[i]], sigma[offset] + 1);
        }
      }
    }
  }
}

__global__ void Update(int *U, int *F, int *sigma, int delta, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  if (offset < size){
    F[offset] = 0;
    if (U[offset] == 1 && sigma[offset] <= delta) {
      U[offset] = 0;
      F[offset] = 1;
    }
  }
}

int Minimum(int *U, int *sigma, int *V, int *E, int num_v, int num_e, int *dev_dest, int *dev_src) {
  uint32_t blocks    = 1;
  uint32_t threads = num_v;


  uint32_t loops;
  uint32_t n_multiple = NearestPowerBase(num_v, threads * blocks * 2, loops);
  uint32_t dev_dest_size = NearestPowerTwo(blocks*loops);

  uint32_t share = NearestPowerTwo(threads);
//  printf("Blocks: %d, Threads:%d\n", blocks, threads);
  reduce_fix<<<blocks, threads, share*sizeof(int)>>>(V, dev_dest, n_multiple,
      share, loops, U, sigma);
  // Recall GPU function: Assumption Destination is power of 2. calculate block
  //                      and threads for each call.
  // GPU Call loop until Threshold
  if (dev_dest_size > 1024) {
    threads = 512;
    blocks = dev_dest_size / threads / 2;
  } else {
    threads = dev_dest_size / 2;
    blocks = 1;
  }

  while (dev_dest_size > 1) {
    int * temp = dev_dest;
    dev_dest = dev_src;
    dev_src = temp;
    reduce<<<blocks, threads, threads*sizeof(int)>>>(dev_src, dev_dest,
      dev_dest_size, U, sigma);
    dev_dest_size = blocks;
    if (dev_dest_size > 1024) {
      threads = 512;
      blocks = dev_dest_size / threads / 2;
    } else {
      threads = dev_dest_size / 2;
      blocks = 1;
    }
  }
  int result;
  cudaMemcpy(&result, dev_dest, sizeof(int), cudaMemcpyDeviceToHost);
  return result;
}
void CudaMallocErrorCheck(void** ptr, int size) {
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    printf("Error: %s", cudaGetErrorString(err));
    exit(1);
  }
}

uint32_t NearestPowerTwo(uint32_t N) {
  uint32_t result = 1;
  while (result < N) {
    result <<= 1;
  }
  return result;
}

uint32_t NearestPowerBase(uint32_t N, uint32_t base, uint32_t &power) {
  uint32_t result = base;
  power = 1;
  while (result < N) {
    result += base;
    power++;
  }
  return result;
}

__global__ void reduce(int *g_idata, int *g_odata, unsigned int n, int *U, int *sigma) {
  // Pointer to shared memory
  extern __shared__ int share_mem[];
  unsigned int thread_id = threadIdx.x;
  unsigned int block_id = blockIdx.x;
  unsigned int block_dim = blockDim.x;
  unsigned int offset = block_id*block_dim*2 + thread_id;

  // Temp result float
  int result = (offset < n && U[offset] == 1) ? g_idata[offset] : INT_MAX;

  // Perform summation
  if (offset + block_dim < n && U[offset + block_dim] == 1)
    result = min(result, g_idata[offset+block_dim]);
  share_mem[thread_id] = result;
  // Sync Threads in a single Block
  __syncthreads();

  // store result to shared memory
  for (unsigned int s=block_dim/2; s>0; s>>=1) {
    if (thread_id < s) {
      share_mem[thread_id] = result = min(result, share_mem[thread_id + s]);
    }
    __syncthreads();
  }

  // Store result to output data pointer
  if (thread_id == 0) g_odata[block_id] = result;
}

__global__ void reduce_fix(int *g_idata, int *g_odata, unsigned int n, unsigned int s_size, unsigned int loops, int *U, int *sigma) {
  // Pointer to shared memory
  extern __shared__ int share_mem[];
  unsigned int thread_id = threadIdx.x;
  for (int i = 0; i < loops; ++i) {
    unsigned int offset = blockIdx.x*blockDim.x*2 + threadIdx.x + blockDim.x * 2 * gridDim.x * i;

    // Temp result float
    int result = (offset < n && U[offset] == 1) ? sigma[offset] : INT_MAX;

    // Perform summation
    if (offset + blockDim.x < n && U[offset + blockDim.x] == 1)
      result = min(result, sigma[offset+blockDim.x]);
    share_mem[thread_id] = result;
//    printf("Result: %d\n", result);
    // Sync Threads in a single Block
    int delta = s_size - blockDim.x;
    if (thread_id + delta > blockDim.x-1) {
      share_mem[thread_id+delta] = INT_MAX;
    }
    __syncthreads();

    // store result to shared memory
    for (unsigned int s=s_size/2; s>0; s>>=1) {
      if (thread_id < s) {
        share_mem[thread_id] = result = min(result, share_mem[thread_id + s]);
      }
      __syncthreads();
    }
    // Store result to output data pointer
    if (thread_id == 0) {
      g_odata[blockIdx.x+ gridDim.x*i] = result;
    }
  }
}
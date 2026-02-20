// Todo: Naive gpu version, test cpu against naiive, benchmark cpu against naiive, optimise, repeat
#include "sw_gpu.cuh"
#include <cstdio>

__constant__ int8_t cuda_blosum62[24 * 24];

__global__ void align(int* scores, const unsigned char* query, const unsigned char* db, const int* offsets, int16_t* H, int16_t* E, int16_t* F, const int M, const int num_alignments, const int db_len){
  int gindex = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index

  // Bounds check
  if (gindex < num_alignments){
    // Calculate length of matrix to fill
    int end_ptr = (gindex + 1) == num_alignments ? db_len : offsets[gindex+1];
    const int N = end_ptr - offsets[gindex] + 1; // Add one for zeros column

    int16_t max_score = 0; // Optimal alignment score

    H = H + offsets[gindex] + gindex; // Add one column of zeros per alignment
    E = E + offsets[gindex] + gindex;
    F = F + offsets[gindex] + gindex;

    // Set first row
    for(int i = 0; i < N; i += 1){
      H[i] = 0;
      F[i] = -10000; // Large negative number
    }

    // Set first column
    for(int j = 0; j < M * (num_alignments + db_len); j += (num_alignments + db_len)){
      H[j] = 0;
      E[j] = -10000;
    }

    // Fill matrices
    for(int i = 1; i < M; i++){ // Skip first row
      for(int j = 1; j < N; j++){ // Skip first column
        int index = i * (num_alignments + db_len) + j;
        int left = index - 1;
        int up = index - (num_alignments + db_len);
        int diag = up - 1;

        E[index] = (H[left] - OPEN) > (E[left] - EXTEND) ? (H[left] - OPEN) : (E[left] - EXTEND); // Horizontal gap
        F[index] = (H[up] - OPEN) > (F[up] - EXTEND) ? (H[up] - OPEN) : (F[up] - EXTEND); // Vertical gap

        int db_idx = offsets[gindex] + (j - 1);
        int16_t val = H[diag] + cuda_blosum62[query[i-1] * 24 + db[db_idx]]; // Diag Score

        // Find max of scores
        val = E[index] > val ? E[index] : val;
        val = F[index] > val ? F[index] : val;
        H[index] = (val < 0) ? 0 : val;

        max_score = H[index] > max_score ? H[index] : max_score; // Update best score seen so far
      }
    }

    // Save best score
    scores[gindex] = max_score;
  }
}

int swGPU(std::vector<int> &scores, const std::vector<unsigned char> &query, const std::vector<unsigned char> &db, const std::vector<int> &offsets){
  // Copy substitution matrix to device
  cudaMemcpyToSymbol(cuda_blosum62, blosum62, sizeof(int8_t) * 24 * 24);

  const int NUM_ALIGNMENTS = scores.size();
  const int DB_LEN = db.size();

  // Kernel launch params
  const int THREADS = 128;
  const int BLOCKS = (NUM_ALIGNMENTS + THREADS - 1) / THREADS;

  // Device copies
  int* d_scores, *d_offsets;
  unsigned char* d_query, *d_db;
  int16_t* d_H, *d_E, *d_F;

  const int M = query.size() + 1; // Number of rows in matrix
  const int N = DB_LEN + NUM_ALIGNMENTS; // Need to allow space for columns of zeros between alignments

  const int scores_bytes = sizeof(int) * NUM_ALIGNMENTS;
  const int offsets_bytes = scores_bytes;
  const int db_bytes = sizeof(unsigned char) * DB_LEN;
  const int query_bytes = sizeof(unsigned char) * query.size();

  // Alloc space for device copies
  cudaMalloc((void**)&d_scores, scores_bytes);
  cudaMalloc((void**)&d_offsets, offsets_bytes);
  cudaMalloc((void**)&d_db, db_bytes);
  cudaMalloc((void**)&d_query, query_bytes);
  cudaMalloc((void**)&d_H, sizeof(int16_t) * M * N);
  cudaMalloc((void**)&d_E, sizeof(int16_t) * M * N);
  cudaMalloc((void**)&d_F, sizeof(int16_t) * M * N);

  // Copy arrays to device
  cudaMemcpy(d_offsets, offsets.data(), offsets_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_db, db.data(), db_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_query, query.data(), query_bytes, cudaMemcpyHostToDevice);

  align<<<BLOCKS, THREADS>>>(d_scores, d_query, d_db, d_offsets, d_H, d_E, d_F, M, NUM_ALIGNMENTS, db.size());

  // Copy result back to host
  cudaMemcpy(scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_scores); cudaFree(d_query); cudaFree(d_db); cudaFree(d_offsets);
  cudaFree(d_H); cudaFree(d_E); cudaFree(d_F);

  return 0;
}
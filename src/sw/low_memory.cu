#include "sw_gpu.cuh"
#include <cstdio>

#define CUDA_CHECK(call)                                                                           \
  do                                                                                               \
  {                                                                                                \
    cudaError_t err = call;                                                                        \
    if (err != cudaSuccess)                                                                        \
    {                                                                                              \
      fprintf(stderr, "CUDA Error:\n  File: %s\n  Line: %d\n  Error code: %d\n  Error text: %s\n", \
              __FILE__, __LINE__, err, cudaGetErrorString(err));                                   \
      goto cleanup;                                                                                \
    }                                                                                              \
  } while (0)

__constant__ int8_t cuda_blosum62[24 * 24];

__global__ void align(int *scores, const unsigned char *query_seq, const unsigned char *db_residues, const int *db_offsets, int16_t *H, int16_t *E, int16_t *F, const int NUM_ROWS, const int NUM_COLS_GLOBAL, const int NUM_RESIDUES)
{
  int gindex = blockIdx.x * blockDim.x + threadIdx.x;

  if (gindex < NUM_COLS_GLOBAL - NUM_RESIDUES)
  {
    const int NUM_COLUMNS_LOCAL = db_offsets[gindex + 1] - db_offsets[gindex] + 1;
    int16_t max_score = 0;

    // Advance pointers to the start of this thread's alignment chunk
    H = H + db_offsets[gindex] + gindex;
    E = E + db_offsets[gindex] + gindex;
    F = F + db_offsets[gindex] + gindex;

    // Set first row (Row 0)
    for (int i = 0; i < NUM_COLUMNS_LOCAL; i++)
    {
      H[i] = 0;
      F[i] = -10000;
    }

    // Set first column for Row 0
    H[0] = 0;
    E[0] = -10000;

    // Fill matrices
    for (int i = 1; i < NUM_ROWS; i++)
    {
      // Calculate our flip-flop rows
      int curr_row = i % 2;
      int prev_row = (1 - curr_row); // If curr is 1, prev is 0. If curr is 0, prev is 1.

      // Initialize the first column for the CURRENT row
      H[curr_row * NUM_COLS_GLOBAL] = 0;
      E[curr_row * NUM_COLS_GLOBAL] = -10000;

      for (int j = 1; j < NUM_COLUMNS_LOCAL; j++)
      {
        // Calculate 1D array indices based on our 2-row buffer
        int index_curr = curr_row * NUM_COLS_GLOBAL + j;
        int index_prev = prev_row * NUM_COLS_GLOBAL + j;

        int left = index_curr - 1;
        int up = index_prev;
        int diag = index_prev - 1;

        // The core logic remains exactly the same, just using new indices
        E[index_curr] = (H[left] - OPEN) > (E[left] - EXTEND) ? (H[left] - OPEN) : (E[left] - EXTEND);
        F[index_curr] = (H[up] - OPEN) > (F[up] - EXTEND) ? (H[up] - OPEN) : (F[up] - EXTEND);

        int db_residue_idx = db_offsets[gindex] + (j - 1);
        int16_t val = H[diag] + cuda_blosum62[query_seq[i - 1] * 24 + db_residues[db_residue_idx]];

        val = E[index_curr] > val ? E[index_curr] : val;
        val = F[index_curr] > val ? F[index_curr] : val;
        H[index_curr] = (val < 0) ? 0 : val;

        max_score = H[index_curr] > max_score ? H[index_curr] : max_score;
      }
    }

    scores[gindex] = max_score;
  }
}

int swGPU(std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets)
{
  const int NUM_ALIGNMENTS = scores.size();
  const int NUM_RESIDUES = db_residues.size();

  // size_t free_mem, total_mem;
  // cudaMemGetInfo(&free_mem, &total_mem);
  // size_t safe_budget = free_mem * 0.9;
  // std::printf("Total VRAM: %ld\n", total_mem);
  // std::printf("Free VRAM: %ld\n", free_mem);
  // std::printf("Safe budget: %ld\n", safe_budget);
  // Total VRAM: 6086262784
  // Free VRAM: 5834276864
  // Safe budget: 5250849177

  // Device copies
  int *d_scores, *d_db_offsets;
  unsigned char *d_query_seq, *d_db_residues;
  int16_t *d_H, *d_E, *d_F;

  const int NUM_COLS = NUM_RESIDUES + NUM_ALIGNMENTS;
  const int NUM_ROWS = query_seq.size() + 1;

  const int scores_bytes = sizeof(int) * NUM_ALIGNMENTS;
  const int offsets_bytes = scores_bytes + sizeof(int);
  const int db_residues_bytes = sizeof(unsigned char) * NUM_RESIDUES;
  const int query_seq_bytes = sizeof(unsigned char) * query_seq.size();
  const int matrix_bytes = 2 * sizeof(int16_t) * NUM_COLS;

  // Kernel launch params
  const int THREADS = 64;
  const int BLOCKS = (NUM_ALIGNMENTS + THREADS - 1) / THREADS;

  int return_code = 0;
  // Copy substitution matrix to device
  CUDA_CHECK(cudaMemcpyToSymbol(cuda_blosum62, blosum62, sizeof(int8_t) * 24 * 24));

  // Alloc space for device copies
  CUDA_CHECK(cudaMalloc((void **)&d_scores, scores_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_db_offsets, offsets_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_db_residues, db_residues_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_query_seq, query_seq_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_H, matrix_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_E, matrix_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_F, matrix_bytes));

  // Copy arrays to device
  CUDA_CHECK(cudaMemcpy(d_db_offsets, db_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_db_residues, db_residues.data(), db_residues_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query_seq, query_seq.data(), query_seq_bytes, cudaMemcpyHostToDevice));

  align<<<BLOCKS, THREADS>>>(d_scores, d_query_seq, d_db_residues, d_db_offsets, d_H, d_E, d_F, NUM_ROWS, NUM_COLS, NUM_RESIDUES);

  CUDA_CHECK(cudaGetLastError());      // Check for valid launch params
  CUDA_CHECK(cudaDeviceSynchronize()); // Check for execution errors (segfaults)

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost));

  goto cleanup_success;

cleanup:
  return_code = -1;

cleanup_success:
  cudaFree(d_scores);
  cudaFree(d_db_offsets);
  cudaFree(d_query_seq);
  cudaFree(d_db_residues);
  cudaFree(d_H);
  cudaFree(d_E);
  cudaFree(d_F);

  return return_code;
}
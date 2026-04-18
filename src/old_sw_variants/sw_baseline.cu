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
  int gindex = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index

  // Bounds check
  if (gindex < NUM_COLS_GLOBAL - NUM_RESIDUES) // Equal to NUM_ALIGNMENTS, saves passing it as a kernel parameter
  {
    // Calculate length of local matrix to fill
    const int NUM_COLUMNS_LOCAL = db_offsets[gindex + 1] - db_offsets[gindex] + 1; // Add one for zeros column

    int16_t max_score = 0; // Optimal alignment score

    H = H + db_offsets[gindex] + gindex; // Add one column of zeros per alignment
    E = E + db_offsets[gindex] + gindex;
    F = F + db_offsets[gindex] + gindex;

    // Set first row
    for (int i = 0; i < NUM_COLUMNS_LOCAL; i += 1)
    {
      H[i] = 0;
      F[i] = -10000; // Large negative number
    }

    // Set first column
    for (int j = 0; j < NUM_ROWS * NUM_COLS_GLOBAL; j += NUM_COLS_GLOBAL)
    {
      H[j] = 0;
      E[j] = -10000;
    }

    // Fill matrices
    for (int i = 1; i < NUM_ROWS; i++)
    { // Skip first row
      for (int j = 1; j < NUM_COLUMNS_LOCAL; j++)
      { // Skip first column
        int index = i * NUM_COLS_GLOBAL + j;
        int left = index - 1;
        int up = index - NUM_COLS_GLOBAL;
        int diag = up - 1;

        E[index] = (H[left] - OPEN) > (E[left] - EXTEND) ? (H[left] - OPEN) : (E[left] - EXTEND); // Horizontal gap
        F[index] = (H[up] - OPEN) > (F[up] - EXTEND) ? (H[up] - OPEN) : (F[up] - EXTEND);         // Vertical gap

        int db_residue_idx = db_offsets[gindex] + (j - 1);
        int16_t val = H[diag] + cuda_blosum62[query_seq[i - 1] * 24 + db_residues[db_residue_idx]]; // Diag Score

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

  const int NUM_ROWS = query_seq.size() + 1;          // Number of rows in matrix
  const int NUM_COLS = NUM_RESIDUES + NUM_ALIGNMENTS; // Need to allow space for columns of zeros between alignments

  const int scores_bytes = sizeof(int) * NUM_ALIGNMENTS;
  const int offsets_bytes = scores_bytes + sizeof(int);
  const int db_residues_bytes = sizeof(unsigned char) * NUM_RESIDUES;
  const int query_seq_bytes = sizeof(unsigned char) * query_seq.size();
  const int matrix_bytes = sizeof(int16_t) * NUM_ROWS * NUM_COLS;
  // std::printf("Matrix bytes: %d\n", matrix_bytes);

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
#include "interAlignGPU.cuh"
#include "params.h"
#include "blosum62.h"
#include <cstdio>
#include <vector>

// Store the query sequence and substitution matrix in constant memory for fast reads
__constant__ unsigned char cuda_query_seq[MAX_QUERY_LEN];

// Terminate if we get any errors from the device
#define CUDA_CHECK(call)                                                                                 \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t err = call;                                                                          \
        if (err != cudaSuccess)                                                                          \
        {                                                                                                \
            fprintf(stderr, "CUDA Error:\n  File: %s\n  Line: %d\n  Error code: %d\n  Error text: %s\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));                                   \
            goto cleanup;                                                                                \
        }                                                                                                \
    } while (0)

// This kernel computes the pairwise global alignment score for every query/target sequence in the database
__global__ void align_nw(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets,
    const int *__restrict__ batch_H_offsets,
    int16_t *__restrict__ H, int16_t *__restrict__ F,
    int8_t *__restrict__ blosum62,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    // Load substitution matrix into fast shared memory
    // Using shared memory is faster than constant memory in this case since
    // Reads are scattered between threads
    __shared__ int8_t shared_blosum62[24 * 24];
    for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x)
        shared_blosum62[i] = blosum62[i];

    __syncthreads(); // Ensure threads matrix is fully loaded

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (gindex >= NUM_ALIGNMENTS)
        return;

    int lane = threadIdx.x % 32;
    int batch_idx = gindex / 32;
    int seq_len = db_offsets[gindex + 1] - db_offsets[gindex];

    int batch_res_start = batch_res_offsets[batch_idx];
    int batch_H_start = batch_H_offsets[batch_idx];

    int batch_res_len = (batch_res_offsets[batch_idx + 1] - batch_res_start) / 32;
    int H_stride = (batch_res_len + 1) * 32;

    int16_t *H_batch = H + batch_H_start;
    int16_t *F_batch = F + batch_H_start;

    // Initialize top-left corner
    H_batch[lane] = 0;
    F_batch[lane] = -10000;

    // Set top Row of matrices
    for (int j = 1; j <= batch_res_len; j++)
    {
        H_batch[j * 32 + lane] = -OPEN - (j - 1) * EXTEND;
        F_batch[j * 32 + lane] = -10000;
    }

    // Used inside the loop to calculate the score of current cell, but declared
    // out here to avoid reading from global memory later when setting the final score
    int16_t h_curr = 0;

    for (int i = 1; i < NUM_ROWS; i++)
    {
        // curr_row alternates between 0 and 1
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        // Initialize first column
        int16_t h_left = -OPEN - (i - 1) * EXTEND;
        int16_t e_left = -10000;
        int16_t h_diag = H_batch[prev_row * H_stride + lane];

        // Update H
        H_batch[curr_row * H_stride + lane] = h_left;

        unsigned char query_char = cuda_query_seq[i - 1];

        // Fill matrix row
        for (int j = 1; j <= seq_len; j++)
        {
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int16_t h_up = H_batch[index_prev];
            int16_t f_up = F_batch[index_prev];

            int16_t e_curr = (h_left - OPEN) > (e_left - EXTEND) ? (h_left - OPEN) : (e_left - EXTEND);
            int16_t f_curr = (h_up - OPEN) > (f_up - EXTEND) ? (h_up - OPEN) : (f_up - EXTEND);

            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int16_t val = h_diag + shared_blosum62[query_char * 24 + db_res];

            val = e_curr > val ? e_curr : val;
            h_curr = f_curr > val ? f_curr : val;

            H_batch[index_curr] = h_curr;
            F_batch[index_curr] = f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    // h_curr contains the final cell value
    scores[gindex] = h_curr;
}

// This kernel computes the pairwise local alignment score for every query/target sequence in the database
__global__ void align_sw(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets,
    const int *__restrict__ batch_H_offsets,
    int16_t *__restrict__ H, int16_t *__restrict__ F,
    int8_t *__restrict__ blosum62,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    // Load substitution matrix into fast shared memory
    // Using shared memory is faster than constant mamory in this case since
    // Reads are scattered between threads
    __shared__ int8_t shared_blosum62[24 * 24];
    for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x)
        shared_blosum62[i] = blosum62[i];

    __syncthreads(); // Ensure threads matrix is fully loaded

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (gindex >= NUM_ALIGNMENTS)
        return;

    int lane = threadIdx.x % 32;
    int batch_idx = gindex / 32;
    int seq_len = db_offsets[gindex + 1] - db_offsets[gindex];

    int batch_res_start = batch_res_offsets[batch_idx];
    int batch_H_start = batch_H_offsets[batch_idx];

    int batch_res_len = (batch_res_offsets[batch_idx + 1] - batch_res_start) / 32;
    int H_stride = (batch_res_len + 1) * 32;

    int16_t *H_batch = H + batch_H_start;
    int16_t *F_batch = F + batch_H_start;

    int16_t max_score = 0; // Local max tracker for SW

    // Initialize Row 0 (including Col 0)
    for (int j = 0; j <= batch_res_len; j++)
    {
        H_batch[j * 32 + lane] = 0;
        F_batch[j * 32 + lane] = -10000;
    }

    for (int i = 1; i < NUM_ROWS; i++)
    {
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        // Initialize First Column
        H_batch[curr_row * H_stride + lane] = 0;

        int16_t h_left = 0;
        int16_t e_left = -10000;
        int16_t h_diag = H_batch[prev_row * H_stride + lane];

        unsigned char query_char = cuda_query_seq[i - 1];

        for (int j = 1; j <= seq_len; j++)
        {
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int16_t h_up = H_batch[index_prev];
            int16_t f_up = F_batch[index_prev];

            int16_t e_curr = (h_left - OPEN) > (e_left - EXTEND) ? (h_left - OPEN) : (e_left - EXTEND);
            int16_t f_curr = (h_up - OPEN) > (f_up - EXTEND) ? (h_up - OPEN) : (f_up - EXTEND);

            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int16_t val = h_diag + shared_blosum62[query_char * 24 + db_res];

            val = e_curr > val ? e_curr : val;
            val = f_curr > val ? f_curr : val;
            int16_t h_curr = (val < 0) ? 0 : val; // SW floor at 0

            max_score = h_curr > max_score ? h_curr : max_score; // Update highest score seen

            H_batch[index_curr] = h_curr;
            F_batch[index_curr] = f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    scores[gindex] = max_score; // Write out the max score found during the trace
}

int interAlignGPU(const int algorithm, std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets)
{
    const int NUM_ALIGNMENTS = db_offsets.size() - 1;
    const int NUM_ROWS = query_seq.size() + 1;

    // Calculate SoA Batch Offsets
    int num_batches = (NUM_ALIGNMENTS + 31) / 32;
    std::vector<int> batch_res_offsets(num_batches + 1, 0);
    std::vector<int> batch_H_offsets(num_batches + 1, 0);

    for (int b = 0; b < num_batches; ++b)
    {
        // Get length of longest sequence in batch
        int max_len = 0;
        for (int t = 0; t < 32; ++t)
        {
            int seq_idx = b * 32 + t;
            if (seq_idx < NUM_ALIGNMENTS)
            {
                int len = db_offsets[seq_idx + 1] - db_offsets[seq_idx];
                if (len > max_len)
                    max_len = len;
            }
        }
        // Batch size is no_batches * longest_seq_in_batch
        batch_res_offsets[b + 1] = batch_res_offsets[b] + (max_len * 32);
        batch_H_offsets[b + 1] = batch_H_offsets[b] + 2 * (max_len + 1) * 32;
    }

    // Device copies
    int *d_scores{}, *d_db_offsets{}, *d_batch_res_offsets{}, *d_batch_H_offsets{};
    unsigned char *d_db_residues{};
    int16_t *d_H{}, *d_F{};
    int8_t *d_blosum62{};

    // Use the calculated offsets for exact sizing
    const int scores_bytes = sizeof(int) * NUM_ALIGNMENTS;
    const int offsets_bytes = sizeof(int) * db_offsets.size();
    const int batch_res_bytes = sizeof(int) * batch_res_offsets.size();
    const int batch_H_bytes = sizeof(int) * batch_H_offsets.size();

    // The padded sizes
    const int db_residues_bytes = sizeof(unsigned char) * batch_res_offsets.back();
    const int matrix_bytes = sizeof(int16_t) * batch_H_offsets.back();

    // Kernel launch params
    const int THREADS = 64;
    const int BLOCKS = (NUM_ALIGNMENTS + THREADS - 1) / THREADS;

    int return_code = 0;

    CUDA_CHECK(cudaMemcpyToSymbol(cuda_query_seq, query_seq.data(), sizeof(unsigned char) * query_seq.size()));

    // Alloc space for device copies
    CUDA_CHECK(cudaMalloc((void **)&d_scores, scores_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_db_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_res_offsets, batch_res_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_H_offsets, batch_H_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_db_residues, db_residues_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_H, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_F, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_blosum62, 24 * 24));

    // Copy arrays to device
    CUDA_CHECK(cudaMemcpy(d_db_offsets, db_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_res_offsets, batch_res_offsets.data(), batch_res_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_H_offsets, batch_H_offsets.data(), batch_H_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db_residues, db_residues.data(), db_residues_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_blosum62, blosum62, sizeof(int8_t) * 24 * 24, cudaMemcpyHostToDevice));

    if (algorithm == 0)
        align_nw<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H, d_F, d_blosum62, NUM_ROWS, NUM_ALIGNMENTS);
    else
        align_sw<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H, d_F, d_blosum62, NUM_ROWS, NUM_ALIGNMENTS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost));

    goto cleanup_success;

cleanup:
    return_code = -1;

cleanup_success:
    cudaFree(d_scores);
    cudaFree(d_db_offsets);
    cudaFree(d_batch_res_offsets);
    cudaFree(d_batch_H_offsets);
    cudaFree(d_db_residues);
    cudaFree(d_H);
    cudaFree(d_F);

    return return_code;
}
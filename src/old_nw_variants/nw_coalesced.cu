#include "sw_gpu.cuh"
#include <cstdio>

#define MAX_QUERY_LEN 1024
__constant__ unsigned char cuda_query_seq[MAX_QUERY_LEN];
__constant__ int8_t cuda_blosum62[24 * 24];

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

__global__ void align_nw(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets, // NEW: Where each batch starts in db_residues
    const int *__restrict__ batch_H_offsets,   // NEW: Where each batch starts in H and F
    int16_t *__restrict__ H, int16_t *__restrict__ F,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;

    // Bail out if we are out of bounds
    if (gindex >= NUM_ALIGNMENTS)
        return;

    // Calculate our place within the warp (lane) and which batch we belong to
    int lane = threadIdx.x % 32;
    int batch_idx = gindex / 32;

    // True sequence length
    int seq_len = db_offsets[gindex + 1] - db_offsets[gindex];

    // Find where our batch starts in global memory
    int batch_res_start = batch_res_offsets[batch_idx];
    int batch_H_start = batch_H_offsets[batch_idx];

    // Calculate max length of this batch to define our memory stride
    int batch_res_len = (batch_res_offsets[batch_idx + 1] - batch_res_start) / 32;
    int H_stride = (batch_res_len + 1) * 32; // +1 because DP matrix has a gap column

    // Advance pointers to the start of this batch's H and F block
    int16_t *H_batch = H + batch_H_start;
    int16_t *F_batch = F + batch_H_start;

    // Initialize Top-Left corner (Row 0, Col 0)
    H_batch[0 * H_stride + 0 * 32 + lane] = 0;
    F_batch[0 * H_stride + 0 * 32 + lane] = -10000;

    // Initialize First Row (Row 0)
    for (int j = 1; j <= batch_res_len; j++)
    {
        H_batch[0 * H_stride + j * 32 + lane] = -OPEN - (j - 1) * EXTEND;
        F_batch[0 * H_stride + j * 32 + lane] = -10000;
    }

    // Fill matrices
    for (int i = 1; i < NUM_ROWS; i++)
    {
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        int16_t h_left = -OPEN - (i - 1) * EXTEND;
        int16_t e_left = -10000;
        int16_t h_diag = H_batch[prev_row * H_stride + 0 * 32 + lane];

        // Initialize the first column for the CURRENT row
        H_batch[curr_row * H_stride + 0 * 32 + lane] = h_left;

        unsigned char query_char = cuda_query_seq[i - 1];

        // Only compute up to our specific sequence length!
        // Inactive threads in the warp will just idle here.
        for (int j = 1; j <= seq_len; j++)
        {
            // Fully coalesced 1D indices
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int16_t h_up = H_batch[index_prev];
            int16_t f_up = F_batch[index_prev];

            int16_t e_curr = (h_left - OPEN) > (e_left - EXTEND) ? (h_left - OPEN) : (e_left - EXTEND);
            int16_t f_curr = (h_up - OPEN) > (f_up - EXTEND) ? (h_up - OPEN) : (f_up - EXTEND);

            // Fully coalesced DB residue fetch
            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int16_t val = h_diag + cuda_blosum62[query_char * 24 + db_res];

            val = e_curr > val ? e_curr : val;
            int16_t h_curr = f_curr > val ? f_curr : val;

            H_batch[index_curr] = h_curr;
            F_batch[index_curr] = f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    int final_row = (NUM_ROWS - 1) % 2;
    scores[gindex] = H_batch[final_row * H_stride + seq_len * 32 + lane];
}

int nwGPU(std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets)
{
    const int NUM_ALIGNMENTS = scores.size();
    const int NUM_ROWS = query_seq.size() + 1;

    // Calculate SoA Batch Offsets
    int num_batches = (NUM_ALIGNMENTS + 31) / 32;
    std::vector<int> batch_res_offsets(num_batches + 1, 0);
    std::vector<int> batch_H_offsets(num_batches + 1, 0);

    for (int b = 0; b < num_batches; ++b)
    {
        int max_len = 0;
        // Find max sequence length in this warp
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
        // Offset for db_residues
        batch_res_offsets[b + 1] = batch_res_offsets[b] + (max_len * 32);
        // Offset for H and F matrices (2 rows, max_len + 1 columns)
        batch_H_offsets[b + 1] = batch_H_offsets[b] + 2 * (max_len + 1) * 32;
    }

    // Device pointers
    int *d_scores, *d_db_offsets, *d_batch_res_offsets, *d_batch_H_offsets;
    unsigned char *d_db_residues;
    int16_t *d_H, *d_F;

    // Byte sizes
    const int scores_bytes = sizeof(int) * NUM_ALIGNMENTS;
    const int offsets_bytes = sizeof(int) * db_offsets.size();
    const int batch_res_bytes = sizeof(int) * batch_res_offsets.size();
    const int batch_H_bytes = sizeof(int) * batch_H_offsets.size();

    const int db_residues_bytes = sizeof(unsigned char) * batch_res_offsets.back();
    const int matrix_bytes = sizeof(int16_t) * batch_H_offsets.back();

    int return_code = 0;

    // Kernel launch params
    const int THREADS = 64;
    const int BLOCKS = (NUM_ALIGNMENTS + THREADS - 1) / THREADS;

    CUDA_CHECK(cudaMemcpyToSymbol(cuda_blosum62, blosum62, sizeof(int8_t) * 24 * 24));
    CUDA_CHECK(cudaMemcpyToSymbol(cuda_query_seq, query_seq.data(), sizeof(unsigned char) * query_seq.size()));

    // Allocate memory
    CUDA_CHECK(cudaMalloc((void **)&d_scores, scores_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_db_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_res_offsets, batch_res_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_H_offsets, batch_H_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_db_residues, db_residues_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_H, matrix_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_F, matrix_bytes));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_db_offsets, db_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_res_offsets, batch_res_offsets.data(), batch_res_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_H_offsets, batch_H_offsets.data(), batch_H_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db_residues, db_residues.data(), db_residues_bytes, cudaMemcpyHostToDevice));

    align_nw<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H, d_F, NUM_ROWS, NUM_ALIGNMENTS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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
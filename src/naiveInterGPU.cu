#include <cstdio>
#include <vector>
#include <algorithm>
#include "params.h"
#include "blosum62.h"

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

__constant__ int8_t cuda_blosum62[24 * 24];

__global__ void align_nw(int *scores, const unsigned char *query_seq, const unsigned char *db_residues, const int *db_offsets, int32_t *H, int32_t *E, int32_t *F, const int NUM_ROWS, const int NUM_COLS_BATCH, const int NUM_ALIGNMENTS)
{
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;

    if (gindex < NUM_ALIGNMENTS)
    {
        const int NUM_COLUMNS_LOCAL = db_offsets[gindex + 1] - db_offsets[gindex] + 1;

        H = H + db_offsets[gindex] + gindex;
        E = E + db_offsets[gindex] + gindex;
        F = F + db_offsets[gindex] + gindex;

        H[0] = 0;
        E[0] = NEG_INF_32;
        F[0] = NEG_INF_32;

        for (int j = 1; j < NUM_COLUMNS_LOCAL; j += 1)
        {
            H[j] = -OPEN - (j - 1) * EXTEND;
            F[j] = NEG_INF_32;
        }

        for (int i = 1; i < NUM_ROWS; i++)
        {
            int index = i * NUM_COLS_BATCH;
            H[index] = -OPEN - (i - 1) * EXTEND;
            E[index] = NEG_INF_32;
        }

        int32_t val = 0;
        for (int i = 1; i < NUM_ROWS; i++)
        {
            for (int j = 1; j < NUM_COLUMNS_LOCAL; j++)
            {
                int index = i * NUM_COLS_BATCH + j;
                int left = index - 1;
                int up = index - NUM_COLS_BATCH;
                int diag = up - 1;

                E[index] = (H[left] - OPEN) > (E[left] - EXTEND) ? (H[left] - OPEN) : (E[left] - EXTEND);
                F[index] = (H[up] - OPEN) > (F[up] - EXTEND) ? (H[up] - OPEN) : (F[up] - EXTEND);

                int db_residue_idx = db_offsets[gindex] + (j - 1);
                val = H[diag] + cuda_blosum62[query_seq[i - 1] * 24 + db_residues[db_residue_idx]];

                val = E[index] > val ? E[index] : val;
                val = F[index] > val ? F[index] : val;
                H[index] = val;
            }
        }
        scores[gindex] = val;
    }
}

__global__ void align_sw(int *scores, const unsigned char *query_seq, const unsigned char *db_residues, const int *db_offsets, int32_t *H, int32_t *E, int32_t *F, const int NUM_ROWS, const int NUM_COLS_BATCH, const int NUM_ALIGNMENTS)
{
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;

    if (gindex < NUM_ALIGNMENTS)
    {
        const int NUM_COLUMNS_LOCAL = db_offsets[gindex + 1] - db_offsets[gindex] + 1;
        int32_t max_score = 0;

        H = H + db_offsets[gindex] + gindex;
        E = E + db_offsets[gindex] + gindex;
        F = F + db_offsets[gindex] + gindex;

        for (int i = 0; i < NUM_COLUMNS_LOCAL; i += 1)
        {
            H[i] = 0;
            F[i] = NEG_INF_32;
        }

        for (int j = 0; j < NUM_ROWS * NUM_COLS_BATCH; j += NUM_COLS_BATCH)
        {
            H[j] = 0;
            E[j] = NEG_INF_32;
        }

        for (int i = 1; i < NUM_ROWS; i++)
        {
            for (int j = 1; j < NUM_COLUMNS_LOCAL; j++)
            {
                int index = i * NUM_COLS_BATCH + j;
                int left = index - 1;
                int up = index - NUM_COLS_BATCH;
                int diag = up - 1;

                E[index] = (H[left] - OPEN) > (E[left] - EXTEND) ? (H[left] - OPEN) : (E[left] - EXTEND);
                F[index] = (H[up] - OPEN) > (F[up] - EXTEND) ? (H[up] - OPEN) : (F[up] - EXTEND);

                int db_residue_idx = db_offsets[gindex] + (j - 1);
                int32_t val = H[diag] + cuda_blosum62[query_seq[i - 1] * 24 + db_residues[db_residue_idx]];

                val = E[index] > val ? E[index] : val;
                val = F[index] > val ? F[index] : val;
                H[index] = (val < 0) ? 0 : val;

                max_score = H[index] > max_score ? H[index] : max_score;
            }
        }
        scores[gindex] = max_score;
    }
}

int naiveInterGPU(int algorithm,
                  std::vector<int> &scores,
                  const std::vector<unsigned char> &query_seq,
                  const std::vector<unsigned char> &db_residues,
                  const std::vector<int> &db_offsets)
{
    const int NUM_ALIGNMENTS = scores.size();
    const int NUM_ROWS = query_seq.size() + 1;
    const int NUM_RESIDUES = db_residues.size();

    const size_t MATRIX_MEMORY_BUDGET = static_cast<size_t>(3) * 1024 * 1024 * 1024; // 3 GB
    const size_t bytes_per_col = sizeof(int32_t) * NUM_ROWS;
    const size_t bytes_per_seq = bytes_per_col * 3; // H, E, F combined

    // Dry-run batching to find max allocation requirements
    int batch_start = 0;
    int max_batch_cols = 0;
    int max_batch_alignments = 0;

    while (batch_start < NUM_ALIGNMENTS)
    {
        int batch_cols = 0;
        int batch_end = batch_start;
        while (batch_end < NUM_ALIGNMENTS)
        {
            int seq_len = db_offsets[batch_end + 1] - db_offsets[batch_end];
            int cols = seq_len + 1; // +1 for zero column
            if (static_cast<size_t>(batch_cols + cols) * bytes_per_seq > MATRIX_MEMORY_BUDGET && batch_end > batch_start)
                break;
            batch_cols += cols;
            batch_end++;
        }
        if (batch_cols > max_batch_cols)
            max_batch_cols = batch_cols;
        if (batch_end - batch_start > max_batch_alignments)
            max_batch_alignments = batch_end - batch_start;
        batch_start = batch_end;
    }

    // Allocate ALL device memory exactly once
    int32_t *d_H{}, *d_E{}, *d_F{};
    unsigned char *d_query_seq{}, *d_db_residues{};
    int *d_scores{}, *d_db_offsets{};
    int return_code = 0;

    CUDA_CHECK(cudaMemcpyToSymbol(cuda_blosum62, blosum62, sizeof(int8_t) * 24 * 24));

    CUDA_CHECK(cudaMalloc((void **)&d_H, max_batch_cols * bytes_per_col));
    CUDA_CHECK(cudaMalloc((void **)&d_E, max_batch_cols * bytes_per_col));
    CUDA_CHECK(cudaMalloc((void **)&d_F, max_batch_cols * bytes_per_col));

    CUDA_CHECK(cudaMalloc((void **)&d_query_seq, query_seq.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_query_seq, query_seq.data(), query_seq.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Upload entirely, batches will just use pointer arithmetic to read their slice
    CUDA_CHECK(cudaMalloc((void **)&d_db_residues, NUM_RESIDUES * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_db_residues, db_residues.data(), NUM_RESIDUES * sizeof(unsigned char), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_scores, max_batch_alignments * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_db_offsets, (max_batch_alignments + 1) * sizeof(int)));

    // Process batches
    batch_start = 0;
    while (batch_start < NUM_ALIGNMENTS)
    {
        int batch_cols = 0;
        int batch_end = batch_start;
        while (batch_end < NUM_ALIGNMENTS)
        {
            int seq_len = db_offsets[batch_end + 1] - db_offsets[batch_end];
            int cols = seq_len + 1;
            if (static_cast<size_t>(batch_cols + cols) * bytes_per_seq > MATRIX_MEMORY_BUDGET && batch_end > batch_start)
                break;
            batch_cols += cols;
            batch_end++;
        }

        int batch_num_alignments = batch_end - batch_start;
        int batch_first_residue = db_offsets[batch_start];

        // Rebase offsets relative to this batch
        std::vector<int> batch_offsets(batch_num_alignments + 1);
        for (int i = 0; i <= batch_num_alignments; ++i)
        {
            batch_offsets[i] = db_offsets[batch_start + i] - batch_first_residue;
        }

        CUDA_CHECK(cudaMemcpy(d_db_offsets, batch_offsets.data(), (batch_num_alignments + 1) * sizeof(int), cudaMemcpyHostToDevice));

        const int THREADS = 64;
        const int BLOCKS = (batch_num_alignments + THREADS - 1) / THREADS;

        if (algorithm == 0)
        {
            align_nw<<<BLOCKS, THREADS>>>(d_scores, d_query_seq, d_db_residues + batch_first_residue, d_db_offsets,
                                          d_H, d_E, d_F, NUM_ROWS, batch_cols, batch_num_alignments);
        }
        else
        {
            align_sw<<<BLOCKS, THREADS>>>(d_scores, d_query_seq, d_db_residues + batch_first_residue, d_db_offsets,
                                          d_H, d_E, d_F, NUM_ROWS, batch_cols, batch_num_alignments);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(scores.data() + batch_start, d_scores, batch_num_alignments * sizeof(int), cudaMemcpyDeviceToHost));

        batch_start = batch_end;
    }

    goto cleanup_success;

cleanup:
    return_code = -1;

cleanup_success:
    if (d_H)
        cudaFree(d_H);
    if (d_E)
        cudaFree(d_E);
    if (d_F)
        cudaFree(d_F);
    if (d_query_seq)
        cudaFree(d_query_seq);
    if (d_db_residues)
        cudaFree(d_db_residues);
    if (d_scores)
        cudaFree(d_scores);
    if (d_db_offsets)
        cudaFree(d_db_offsets);

    return return_code;
}
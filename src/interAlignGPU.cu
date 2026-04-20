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

// int16 NW kernel with overflow detection. Any cell whose magnitude reaches
// OVERFLOW_THRESHOLD_16 sets overflow_flags[gindex] so the host can re-run that
// alignment's batch through the int32 kernel.
__global__ void align_nw_i16(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets,
    const int *__restrict__ batch_H_offsets,
    int16_t *__restrict__ H, int16_t *__restrict__ F,
    int8_t *__restrict__ blosum62,
    int *__restrict__ overflow_flags,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    __shared__ int8_t shared_blosum62[24 * 24];
    for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x)
        shared_blosum62[i] = blosum62[i];

    __syncthreads();

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

    int overflow = 0;

    H_batch[lane] = 0;
    F_batch[lane] = NEG_INF_16;

    for (int j = 1; j <= batch_res_len; j++)
    {
        int init_val = -OPEN - (j - 1) * EXTEND;
        if (init_val <= -OVERFLOW_THRESHOLD_16)
            overflow = 1;
        H_batch[j * 32 + lane] = (int16_t)init_val;
        F_batch[j * 32 + lane] = NEG_INF_16;
    }

    // Do arithmetic in int32 to avoid mid-computation wrap; flag and cast to
    // int16 on store. Wrapped stores cascade into garbage downstream, but the
    // overflow flag ensures the host recomputes this alignment.
    int h_curr = 0;

    for (int i = 1; i < NUM_ROWS; i++)
    {
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        int h_left_init = -OPEN - (i - 1) * EXTEND;
        if (h_left_init <= -OVERFLOW_THRESHOLD_16)
            overflow = 1;

        int h_left = h_left_init;
        int e_left = NEG_INF_16;
        int h_diag = H_batch[prev_row * H_stride + lane];

        H_batch[curr_row * H_stride + lane] = (int16_t)h_left;

        unsigned char query_char = cuda_query_seq[i - 1];

        for (int j = 1; j <= seq_len; j++)
        {
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int h_up = H_batch[index_prev];
            int f_up = F_batch[index_prev];

            int e_curr = max(h_left - OPEN, e_left - EXTEND);
            int f_curr = max(h_up - OPEN, f_up - EXTEND);

            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int val = h_diag + shared_blosum62[query_char * 24 + db_res];
            val = max(val, e_curr);
            h_curr = max(val, f_curr);

            if (h_curr >= OVERFLOW_THRESHOLD_16 || h_curr <= -OVERFLOW_THRESHOLD_16)
                overflow = 1;

            H_batch[index_curr] = (int16_t)h_curr;
            F_batch[index_curr] = (int16_t)f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    if (overflow)
        overflow_flags[gindex] = 1;

    scores[gindex] = h_curr;
}

// int16 SW kernel with overflow detection. SW scores are non-negative so only
// positive overflow can occur; we still track max in int32 so that a flagged
// alignment's partial result is meaningful if we ever want to inspect it.
__global__ void align_sw_i16(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets,
    const int *__restrict__ batch_H_offsets,
    int16_t *__restrict__ H, int16_t *__restrict__ F,
    int8_t *__restrict__ blosum62,
    int *__restrict__ overflow_flags,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    __shared__ int8_t shared_blosum62[24 * 24];
    for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x)
        shared_blosum62[i] = blosum62[i];

    __syncthreads();

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

    int max_score = 0;
    int overflow = 0;

    for (int j = 0; j <= batch_res_len; j++)
    {
        H_batch[j * 32 + lane] = 0;
        F_batch[j * 32 + lane] = NEG_INF_16;
    }

    for (int i = 1; i < NUM_ROWS; i++)
    {
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        H_batch[curr_row * H_stride + lane] = 0;

        int h_left = 0;
        int e_left = NEG_INF_16;
        int h_diag = H_batch[prev_row * H_stride + lane];

        unsigned char query_char = cuda_query_seq[i - 1];

        for (int j = 1; j <= seq_len; j++)
        {
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int h_up = H_batch[index_prev];
            int f_up = F_batch[index_prev];

            int e_curr = max(h_left - OPEN, e_left - EXTEND);
            int f_curr = max(h_up - OPEN, f_up - EXTEND);

            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int val = h_diag + shared_blosum62[query_char * 24 + db_res];
            val = max(val, e_curr);
            val = max(val, f_curr);
            int h_curr = (val < 0) ? 0 : val;

            if (h_curr >= OVERFLOW_THRESHOLD_16)
                overflow = 1;

            max_score = h_curr > max_score ? h_curr : max_score;

            H_batch[index_curr] = (int16_t)h_curr;
            F_batch[index_curr] = (int16_t)f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    if (overflow)
        overflow_flags[gindex] = 1;

    scores[gindex] = max_score;
}

// int32 NW fallback. If batch_mask is non-null, only process batches whose
// mask entry is set — used to re-run only the batches containing an alignment
// that overflowed in the int16 pass.
__global__ void align_nw(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets,
    const int *__restrict__ batch_H_offsets,
    int32_t *__restrict__ H, int32_t *__restrict__ F,
    int8_t *__restrict__ blosum62,
    const int *__restrict__ batch_mask,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    __shared__ int8_t shared_blosum62[24 * 24];
    for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x)
        shared_blosum62[i] = blosum62[i];

    __syncthreads();

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (gindex >= NUM_ALIGNMENTS)
        return;

    int lane = threadIdx.x % 32;
    int batch_idx = gindex / 32;

    if (batch_mask != nullptr && batch_mask[batch_idx] == 0)
        return;

    int seq_len = db_offsets[gindex + 1] - db_offsets[gindex];

    int batch_res_start = batch_res_offsets[batch_idx];
    int batch_H_start = batch_H_offsets[batch_idx];

    int batch_res_len = (batch_res_offsets[batch_idx + 1] - batch_res_start) / 32;
    int H_stride = (batch_res_len + 1) * 32;

    int32_t *H_batch = H + batch_H_start;
    int32_t *F_batch = F + batch_H_start;

    H_batch[lane] = 0;
    F_batch[lane] = NEG_INF_32;

    for (int j = 1; j <= batch_res_len; j++)
    {
        H_batch[j * 32 + lane] = -OPEN - (j - 1) * EXTEND;
        F_batch[j * 32 + lane] = NEG_INF_32;
    }

    int32_t h_curr = 0;

    for (int i = 1; i < NUM_ROWS; i++)
    {
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        int32_t h_left = -OPEN - (i - 1) * EXTEND;
        int32_t e_left = NEG_INF_32;
        int32_t h_diag = H_batch[prev_row * H_stride + lane];

        H_batch[curr_row * H_stride + lane] = h_left;

        unsigned char query_char = cuda_query_seq[i - 1];

        for (int j = 1; j <= seq_len; j++)
        {
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int32_t h_up = H_batch[index_prev];
            int32_t f_up = F_batch[index_prev];

            int32_t e_curr = max(h_left - OPEN, e_left - EXTEND);
            int32_t f_curr = max(h_up - OPEN, f_up - EXTEND);

            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int32_t val = h_diag + shared_blosum62[query_char * 24 + db_res];

            val = max(e_curr, val);
            h_curr = max(f_curr, val);

            H_batch[index_curr] = h_curr;
            F_batch[index_curr] = f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    scores[gindex] = h_curr;
}

__global__ void align_sw(
    int *__restrict__ scores,
    const unsigned char *__restrict__ db_residues,
    const int *__restrict__ db_offsets,
    const int *__restrict__ batch_res_offsets,
    const int *__restrict__ batch_H_offsets,
    int32_t *__restrict__ H, int32_t *__restrict__ F,
    int8_t *__restrict__ blosum62,
    const int *__restrict__ batch_mask,
    const int NUM_ROWS,
    const int NUM_ALIGNMENTS)
{
    __shared__ int8_t shared_blosum62[24 * 24];
    for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x)
        shared_blosum62[i] = blosum62[i];

    __syncthreads();

    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (gindex >= NUM_ALIGNMENTS)
        return;

    int lane = threadIdx.x % 32;
    int batch_idx = gindex / 32;

    if (batch_mask != nullptr && batch_mask[batch_idx] == 0)
        return;

    int seq_len = db_offsets[gindex + 1] - db_offsets[gindex];

    int batch_res_start = batch_res_offsets[batch_idx];
    int batch_H_start = batch_H_offsets[batch_idx];

    int batch_res_len = (batch_res_offsets[batch_idx + 1] - batch_res_start) / 32;
    int H_stride = (batch_res_len + 1) * 32;

    int32_t *H_batch = H + batch_H_start;
    int32_t *F_batch = F + batch_H_start;

    int32_t max_score = 0;

    for (int j = 0; j <= batch_res_len; j++)
    {
        H_batch[j * 32 + lane] = 0;
        F_batch[j * 32 + lane] = NEG_INF_32;
    }

    for (int i = 1; i < NUM_ROWS; i++)
    {
        int curr_row = i % 2;
        int prev_row = 1 - curr_row;

        H_batch[curr_row * H_stride + lane] = 0;

        int32_t h_left = 0;
        int32_t e_left = NEG_INF_32;
        int32_t h_diag = H_batch[prev_row * H_stride + lane];

        unsigned char query_char = cuda_query_seq[i - 1];

        for (int j = 1; j <= seq_len; j++)
        {
            int index_curr = curr_row * H_stride + j * 32 + lane;
            int index_prev = prev_row * H_stride + j * 32 + lane;

            int32_t h_up = H_batch[index_prev];
            int32_t f_up = F_batch[index_prev];

            int32_t e_curr = max(h_left - OPEN, e_left - EXTEND);
            int32_t f_curr = max(h_up - OPEN, f_up - EXTEND);

            int db_residue_idx = batch_res_start + (j - 1) * 32 + lane;
            unsigned char db_res = db_residues[db_residue_idx];

            int32_t val = h_diag + shared_blosum62[query_char * 24 + db_res];

            val = max(e_curr, val);
            val = max(f_curr, val);
            int32_t h_curr = max(val, 0);

            max_score = max(h_curr, max_score);

            H_batch[index_curr] = h_curr;
            F_batch[index_curr] = f_curr;

            h_left = h_curr;
            e_left = e_curr;
            h_diag = h_up;
        }
    }

    scores[gindex] = max_score;
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
        batch_res_offsets[b + 1] = batch_res_offsets[b] + (max_len * 32);
        batch_H_offsets[b + 1] = batch_H_offsets[b] + 2 * (max_len + 1) * 32;
    }

    // Device copies
    int *d_scores{}, *d_db_offsets{}, *d_batch_res_offsets{}, *d_batch_H_offsets{};
    int *d_overflow_flags{}, *d_batch_mask{};
    unsigned char *d_db_residues{};
    int16_t *d_H16{}, *d_F16{};
    int32_t *d_H32{}, *d_F32{};
    int8_t *d_blosum62{};

    const int scores_bytes = sizeof(int) * NUM_ALIGNMENTS;
    const int offsets_bytes = sizeof(int) * db_offsets.size();
    const int batch_res_bytes = sizeof(int) * batch_res_offsets.size();
    const int batch_H_bytes = sizeof(int) * batch_H_offsets.size();

    const int db_residues_bytes = sizeof(unsigned char) * batch_res_offsets.back();
    const size_t matrix_cells = batch_H_offsets.back();
    const size_t matrix_bytes_16 = sizeof(int16_t) * matrix_cells;
    const size_t matrix_bytes_32 = sizeof(int32_t) * matrix_cells;

    const int THREADS = 64;
    const int BLOCKS = (NUM_ALIGNMENTS + THREADS - 1) / THREADS;

    int return_code = 0;

    CUDA_CHECK(cudaMemcpyToSymbol(cuda_query_seq, query_seq.data(), sizeof(unsigned char) * query_seq.size()));

    CUDA_CHECK(cudaMalloc((void **)&d_scores, scores_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_db_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_res_offsets, batch_res_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_batch_H_offsets, batch_H_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_db_residues, db_residues_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_H16, matrix_bytes_16));
    CUDA_CHECK(cudaMalloc((void **)&d_F16, matrix_bytes_16));
    CUDA_CHECK(cudaMalloc((void **)&d_overflow_flags, sizeof(int) * NUM_ALIGNMENTS));
    CUDA_CHECK(cudaMalloc((void **)&d_blosum62, 24 * 24));

    CUDA_CHECK(cudaMemcpy(d_db_offsets, db_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_res_offsets, batch_res_offsets.data(), batch_res_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_H_offsets, batch_H_offsets.data(), batch_H_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_db_residues, db_residues.data(), db_residues_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blosum62, blosum62, sizeof(int8_t) * 24 * 24, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_overflow_flags, 0, sizeof(int) * NUM_ALIGNMENTS));

    if (algorithm == 0)
        align_nw_i16<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H16, d_F16, d_blosum62, d_overflow_flags, NUM_ROWS, NUM_ALIGNMENTS);
    else
        align_sw_i16<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H16, d_F16, d_blosum62, d_overflow_flags, NUM_ROWS, NUM_ALIGNMENTS);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pull flags back and build a per-batch mask. A batch of 32 lanes shares
    // its SoA layout, so if any alignment in it overflowed we re-run the
    // whole batch with the int32 kernel. Scoped so the goto in CUDA_CHECK
    // never jumps past a non-trivial local.
    {
        std::vector<int> overflow_flags(NUM_ALIGNMENTS);
        CUDA_CHECK(cudaMemcpy(overflow_flags.data(), d_overflow_flags, sizeof(int) * NUM_ALIGNMENTS, cudaMemcpyDeviceToHost));

        // int16 H/F buffers are dead once the kernel returns; free now to cap
        // peak memory before allocating int32 buffers for the fallback.
        cudaFree(d_H16);
        cudaFree(d_F16);
        d_H16 = nullptr;
        d_F16 = nullptr;

        std::vector<int> batch_mask(num_batches, 0);
        int num_overflow_batches = 0;
        for (int g = 0; g < NUM_ALIGNMENTS; g++)
        {
            if (overflow_flags[g])
            {
                int b = g / 32;
                if (batch_mask[b] == 0)
                    num_overflow_batches++;
                batch_mask[b] = 1;
            }
        }

        if (num_overflow_batches > 0)
        {
            CUDA_CHECK(cudaMalloc((void **)&d_H32, matrix_bytes_32));
            CUDA_CHECK(cudaMalloc((void **)&d_F32, matrix_bytes_32));
            CUDA_CHECK(cudaMalloc((void **)&d_batch_mask, sizeof(int) * num_batches));
            CUDA_CHECK(cudaMemcpy(d_batch_mask, batch_mask.data(), sizeof(int) * num_batches, cudaMemcpyHostToDevice));

            if (algorithm == 0)
                align_nw<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H32, d_F32, d_blosum62, d_batch_mask, NUM_ROWS, NUM_ALIGNMENTS);
            else
                align_sw<<<BLOCKS, THREADS>>>(d_scores, d_db_residues, d_db_offsets, d_batch_res_offsets, d_batch_H_offsets, d_H32, d_F32, d_blosum62, d_batch_mask, NUM_ROWS, NUM_ALIGNMENTS);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores_bytes, cudaMemcpyDeviceToHost));
    }

    goto cleanup_success;

cleanup:
    return_code = -1;

cleanup_success:
    cudaFree(d_scores);
    cudaFree(d_db_offsets);
    cudaFree(d_batch_res_offsets);
    cudaFree(d_batch_H_offsets);
    cudaFree(d_db_residues);
    cudaFree(d_H16);
    cudaFree(d_F16);
    cudaFree(d_H32);
    cudaFree(d_F32);
    cudaFree(d_overflow_flags);
    cudaFree(d_batch_mask);
    cudaFree(d_blosum62);

    return return_code;
}

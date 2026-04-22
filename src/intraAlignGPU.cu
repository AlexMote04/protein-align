#include "intraAlignGPU.cuh"
#include "params.h"
#include "blosum62.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>

__constant__ int8_t cuda_blosum62[24 * 24];

#define TILE_SIZE 32

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

// int16 variant with overflow detection. Cells and boundary inits are
// checked against OVERFLOW_THRESHOLD_16; any trip sets d_overflow_flag so the
// host can re-run this sequence through the int32 kernel.
__global__ void
align_massive_wavefront_tile_i16(
    int16_t *__restrict__ row_buf_H, int16_t *__restrict__ col_buf_H,
    int16_t *__restrict__ col_buf_E, int16_t *__restrict__ row_buf_F,
    int16_t *__restrict__ d_corner_H,
    const unsigned char *__restrict__ d_query, const unsigned char *__restrict__ d_target,
    const int query_len, const int target_len,
    const int step, const int num_blocks_x, const int num_blocks_y,
    const int algorithm, int *__restrict__ d_max_score,
    int *__restrict__ d_overflow_flag)
{
    int min_x = max(0, step - num_blocks_y + 1);
    int block_x = min_x + blockIdx.x;
    int block_y = step - block_x;

    int tx = threadIdx.x;

    int global_i = block_x * TILE_SIZE + tx;
    int global_j = block_y * TILE_SIZE + tx;

    __shared__ int16_t H_tile[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ int16_t E_tile[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ int16_t F_tile[TILE_SIZE + 1][TILE_SIZE + 1];

    __shared__ unsigned char q_sub[TILE_SIZE];
    __shared__ unsigned char t_sub[TILE_SIZE];
    __shared__ int local_max;
    __shared__ int local_overflow;

    if (tx == 0)
    {
        local_max = 0;
        local_overflow = 0;
    }

    if (global_i < query_len)
        q_sub[tx] = d_query[global_i];
    if (global_j < target_len)
        t_sub[tx] = d_target[global_j];

    // Top boundary
    if (global_j < target_len)
    {
        if (block_x > 0)
        {
            H_tile[0][tx + 1] = row_buf_H[global_j];
            F_tile[0][tx + 1] = row_buf_F[global_j];
        }
        else
        {
            int init_H = (algorithm == 0) ? (-OPEN - (global_j)*EXTEND) : 0;
            if (init_H <= -OVERFLOW_THRESHOLD_16)
                local_overflow = 1;
            H_tile[0][tx + 1] = (int16_t)init_H;
            F_tile[0][tx + 1] = NEG_INF_16;
        }
    }

    // Left boundary
    if (global_i < query_len)
    {
        if (block_y > 0)
        {
            H_tile[tx + 1][0] = col_buf_H[global_i];
            E_tile[tx + 1][0] = col_buf_E[global_i];
        }
        else
        {
            int init_H = (algorithm == 0) ? (-OPEN - (global_i)*EXTEND) : 0;
            if (init_H <= -OVERFLOW_THRESHOLD_16)
                local_overflow = 1;
            H_tile[tx + 1][0] = (int16_t)init_H;
            E_tile[tx + 1][0] = NEG_INF_16;
        }
    }

    // Top-left corner dependency for cell (1,1) of the tile
    if (tx == 0)
    {
        if (block_x > 0 && block_y > 0)
        {
            H_tile[0][0] = d_corner_H[(block_x - 1) * num_blocks_y + (block_y - 1)];
        }
        else if (algorithm == 0)
        {
            int corner_H = 0;
            if (block_x == 0 && block_y == 0)
                corner_H = 0;
            else if (block_x == 0)
                corner_H = -OPEN - (block_y * TILE_SIZE - 1) * EXTEND;
            else if (block_y == 0)
                corner_H = -OPEN - (block_x * TILE_SIZE - 1) * EXTEND;
            if (corner_H <= -OVERFLOW_THRESHOLD_16)
                local_overflow = 1;
            H_tile[0][0] = (int16_t)corner_H;
        }
        else
        {
            H_tile[0][0] = 0;
        }
    }

    __syncthreads();

    // The Wavefront Execution
    int num_diagonals = 2 * TILE_SIZE - 1;
    int thread_max = 0;

    for (int k = 0; k < num_diagonals; k++)
    {
        int j = tx + 1;
        int i = k - tx + 1;

        if (i >= 1 && i <= TILE_SIZE)
        {
            int seq_i = block_x * TILE_SIZE + i - 1;
            int seq_j = block_y * TILE_SIZE + j - 1;

            if (seq_i < query_len && seq_j < target_len)
            {
                // Compute in int to avoid mid-step wrap. Cast to int16_t on
                // store — wrapped stores are OK because the flag below forces
                // the host to recompute this sequence in int32.
                int h_diag = H_tile[i - 1][j - 1];
                int h_up = H_tile[i - 1][j];
                int h_left = H_tile[i][j - 1];
                int e_left = E_tile[i][j - 1];
                int f_up = F_tile[i - 1][j];

                unsigned char q_char = q_sub[i - 1];
                unsigned char t_char = t_sub[j - 1];

                int match_score = h_diag + cuda_blosum62[q_char * 24 + t_char];

                int e_curr = max(h_left - OPEN, e_left - EXTEND);
                int f_curr = max(h_up - OPEN, f_up - EXTEND);

                int h_curr = max(match_score, max(e_curr, f_curr));

                if (algorithm == 1)
                {
                    h_curr = max(0, h_curr);
                    if (h_curr > thread_max)
                        thread_max = h_curr;
                }

                if (h_curr >= OVERFLOW_THRESHOLD_16 || h_curr <= -OVERFLOW_THRESHOLD_16)
                    local_overflow = 1;

                H_tile[i][j] = (int16_t)h_curr;
                E_tile[i][j] = (int16_t)e_curr;
                F_tile[i][j] = (int16_t)f_curr;
            }
        }
        __syncthreads();
    }

    // Save this tile's bottom-right corner for the tile diagonally below us
    if (tx == 0)
    {
        int last_i = min(TILE_SIZE, query_len - block_x * TILE_SIZE);
        int last_j = min(TILE_SIZE, target_len - block_y * TILE_SIZE);

        d_corner_H[block_x * num_blocks_y + block_y] = H_tile[last_i][last_j];
    }

    int valid_i = min(TILE_SIZE, query_len - block_x * TILE_SIZE);
    int valid_j = min(TILE_SIZE, target_len - block_y * TILE_SIZE);

    if (global_j < target_len)
    {
        row_buf_H[global_j] = H_tile[valid_i][tx + 1];
        row_buf_F[global_j] = F_tile[valid_i][tx + 1];
    }

    if (global_i < query_len)
    {
        col_buf_H[global_i] = H_tile[tx + 1][valid_j];
        col_buf_E[global_i] = E_tile[tx + 1][valid_j];
    }

    if (algorithm == 1)
    {
        atomicMax(&local_max, thread_max);
        __syncthreads();
        if (tx == 0 && local_max > 0)
        {
            atomicMax(d_max_score, local_max);
        }
    }

    if (tx == 0 && local_overflow)
    {
        atomicOr(d_overflow_flag, 1);
    }
}

// int32 fallback kernel — unchanged from the original implementation.
__global__ void
align_massive_wavefront_tile(
    int32_t *__restrict__ row_buf_H, int32_t *__restrict__ col_buf_H,
    int32_t *__restrict__ col_buf_E, int32_t *__restrict__ row_buf_F,
    int32_t *__restrict__ d_corner_H,
    const unsigned char *__restrict__ d_query, const unsigned char *__restrict__ d_target,
    const int query_len, const int target_len,
    const int step, const int num_blocks_x, const int num_blocks_y,
    const int algorithm, int *__restrict__ d_max_score)
{
    int min_x = max(0, step - num_blocks_y + 1);
    int block_x = min_x + blockIdx.x;
    int block_y = step - block_x;

    int tx = threadIdx.x;

    int global_i = block_x * TILE_SIZE + tx;
    int global_j = block_y * TILE_SIZE + tx;

    __shared__ int32_t H_tile[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ int32_t E_tile[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ int32_t F_tile[TILE_SIZE + 1][TILE_SIZE + 1];

    __shared__ unsigned char q_sub[TILE_SIZE];
    __shared__ unsigned char t_sub[TILE_SIZE];
    __shared__ int local_max;

    if (tx == 0)
        local_max = 0;

    if (global_i < query_len)
        q_sub[tx] = d_query[global_i];
    if (global_j < target_len)
        t_sub[tx] = d_target[global_j];

    if (global_j < target_len)
    {
        if (block_x > 0)
        {
            H_tile[0][tx + 1] = row_buf_H[global_j];
            F_tile[0][tx + 1] = row_buf_F[global_j];
        }
        else
        {
            H_tile[0][tx + 1] = (algorithm == 0) ? (-OPEN - (global_j)*EXTEND) : 0;
            F_tile[0][tx + 1] = NEG_INF_32;
        }
    }

    if (global_i < query_len)
    {
        if (block_y > 0)
        {
            H_tile[tx + 1][0] = col_buf_H[global_i];
            E_tile[tx + 1][0] = col_buf_E[global_i];
        }
        else
        {
            H_tile[tx + 1][0] = (algorithm == 0) ? (-OPEN - (global_i)*EXTEND) : 0;
            E_tile[tx + 1][0] = NEG_INF_32;
        }
    }

    if (tx == 0)
    {
        if (block_x > 0 && block_y > 0)
        {
            H_tile[0][0] = d_corner_H[(block_x - 1) * num_blocks_y + (block_y - 1)];
        }
        else if (algorithm == 0)
        {
            if (block_x == 0 && block_y == 0)
                H_tile[0][0] = 0;
            else if (block_x == 0)
                H_tile[0][0] = -OPEN - (block_y * TILE_SIZE - 1) * EXTEND;
            else if (block_y == 0)
                H_tile[0][0] = -OPEN - (block_x * TILE_SIZE - 1) * EXTEND;
        }
        else
        {
            H_tile[0][0] = 0;
        }
    }

    __syncthreads();

    int num_diagonals = 2 * TILE_SIZE - 1;
    int thread_max = 0;

    for (int k = 0; k < num_diagonals; k++)
    {
        int j = tx + 1;
        int i = k - tx + 1;

        if (i >= 1 && i <= TILE_SIZE)
        {
            int seq_i = block_x * TILE_SIZE + i - 1;
            int seq_j = block_y * TILE_SIZE + j - 1;

            if (seq_i < query_len && seq_j < target_len)
            {
                int32_t h_diag = H_tile[i - 1][j - 1];
                int32_t h_up = H_tile[i - 1][j];
                int32_t h_left = H_tile[i][j - 1];
                int32_t e_left = E_tile[i][j - 1];
                int32_t f_up = F_tile[i - 1][j];

                unsigned char q_char = q_sub[i - 1];
                unsigned char t_char = t_sub[j - 1];

                int32_t match_score = h_diag + cuda_blosum62[q_char * 24 + t_char];

                int32_t e_curr = max((int32_t)(h_left - OPEN), (int32_t)(e_left - EXTEND));
                int32_t f_curr = max((int32_t)(h_up - OPEN), (int32_t)(f_up - EXTEND));

                int32_t h_curr = max(match_score, max(e_curr, f_curr));

                if (algorithm == 1)
                {
                    h_curr = max((int32_t)0, h_curr);
                    if (h_curr > thread_max)
                        thread_max = h_curr;
                }

                H_tile[i][j] = h_curr;
                E_tile[i][j] = e_curr;
                F_tile[i][j] = f_curr;
            }
        }
        __syncthreads();
    }

    if (tx == 0)
    {
        int last_i = min(TILE_SIZE, query_len - block_x * TILE_SIZE);
        int last_j = min(TILE_SIZE, target_len - block_y * TILE_SIZE);

        d_corner_H[block_x * num_blocks_y + block_y] = H_tile[last_i][last_j];
    }

    int valid_i = min(TILE_SIZE, query_len - block_x * TILE_SIZE);
    int valid_j = min(TILE_SIZE, target_len - block_y * TILE_SIZE);

    if (global_j < target_len)
    {
        row_buf_H[global_j] = H_tile[valid_i][tx + 1];
        row_buf_F[global_j] = F_tile[valid_i][tx + 1];
    }

    if (global_i < query_len)
    {
        col_buf_H[global_i] = H_tile[tx + 1][valid_j];
        col_buf_E[global_i] = E_tile[tx + 1][valid_j];
    }

    if (algorithm == 1)
    {
        atomicMax(&local_max, thread_max);
        __syncthreads();
        if (tx == 0 && local_max > 0)
        {
            atomicMax(d_max_score, local_max);
        }
    }
}

float intraAlignGPU(
    int algorithm,
    const std::vector<unsigned char> &query_seq,
    const std::vector<std::vector<unsigned char>> &large_seqs,
    std::vector<int> &results,
    int result_offset)
{
    if (large_seqs.empty())
        return 0;

    const int query_len = query_seq.size();
    const int max_target_len = large_seqs.back().size(); // Last element is always largest since db gets sorted
    const size_t num_large = large_seqs.size();

    const int max_num_blocks_x = (query_len + TILE_SIZE - 1) / TILE_SIZE;
    const int max_num_blocks_y = (max_target_len + TILE_SIZE - 1) / TILE_SIZE;
    const int max_num_tiles_total = max_num_blocks_x * max_num_blocks_y;

    constexpr int MAX_STREAMS = 4;
    const int num_streams = std::min<int>(MAX_STREAMS, static_cast<int>(num_large));

    cudaStream_t streams[MAX_STREAMS] = {};
    bool stream_created[MAX_STREAMS] = {};

    unsigned char *d_query = nullptr;
    unsigned char *d_target[MAX_STREAMS] = {};
    int16_t *d_row_H[MAX_STREAMS] = {};
    int16_t *d_row_F[MAX_STREAMS] = {};
    int16_t *d_col_H[MAX_STREAMS] = {};
    int16_t *d_col_E[MAX_STREAMS] = {};
    int16_t *d_corner_H[MAX_STREAMS] = {};
    int *d_max_score[MAX_STREAMS] = {};
    int *d_overflow_flag[MAX_STREAMS] = {};
    unsigned char *h_pinned_target[MAX_STREAMS] = {};

    // int32 fallback buffers — shared (not per-stream) and allocated only if
    // some sequence overflows. Fallback runs are serialized on the default
    // stream; in the common case (no overflow) this memory is never spent.
    int32_t *d_row_H32 = nullptr;
    int32_t *d_row_F32 = nullptr;
    int32_t *d_col_H32 = nullptr;
    int32_t *d_col_E32 = nullptr;
    int32_t *d_corner_H32 = nullptr;
    int *d_max_score32 = nullptr;
    unsigned char *d_target_fb = nullptr;

    int16_t *h_pinned_nw = nullptr;
    int *h_pinned_sw = nullptr;
    int *h_pinned_overflow = nullptr;

    cudaEvent_t ev_start[MAX_STREAMS] = {};
    cudaEvent_t ev_stop[MAX_STREAMS] = {};
    bool ev_created[MAX_STREAMS] = {};
    bool needs_timing[MAX_STREAMS] = {};
    cudaEvent_t ev_fb_start = nullptr, ev_fb_stop = nullptr;
    float kernel_ms = 0.0f;
    float return_ms = -1.0f;

    // Shared read-only device data
    CUDA_CHECK(cudaMalloc((void **)&d_query, query_len * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_query, query_seq.data(), query_len * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(cuda_blosum62, blosum62, sizeof(int8_t) * 24 * 24));

    CUDA_CHECK(cudaMallocHost((void **)&h_pinned_overflow, num_large * sizeof(int)));
    if (algorithm == 0)
        CUDA_CHECK(cudaMallocHost((void **)&h_pinned_nw, num_large * sizeof(int16_t)));
    else
        CUDA_CHECK(cudaMallocHost((void **)&h_pinned_sw, num_large * sizeof(int)));

    // Per-stream int16 buffers
    for (int s = 0; s < num_streams; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        stream_created[s] = true;
        CUDA_CHECK(cudaMalloc((void **)&d_target[s], max_target_len * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc((void **)&d_row_H[s], max_target_len * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_row_F[s], max_target_len * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_col_H[s], query_len * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_col_E[s], query_len * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_corner_H[s], max_num_tiles_total * sizeof(int16_t)));
        CUDA_CHECK(cudaMalloc((void **)&d_max_score[s], sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&d_overflow_flag[s], sizeof(int)));
        CUDA_CHECK(cudaMallocHost((void **)&h_pinned_target[s], max_target_len * sizeof(unsigned char)));
        CUDA_CHECK(cudaEventCreate(&ev_start[s]));
        CUDA_CHECK(cudaEventCreate(&ev_stop[s]));
        ev_created[s] = true;
    }

    // Dispatch int16 kernels, rotating through streams to overlap H2D, compute, and D2H
    for (size_t i = 0; i < num_large; i++)
    {
        const int s = static_cast<int>(i % num_streams);
        const auto &target_seq = large_seqs[i];
        const int target_len = target_seq.size();

        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        if (needs_timing[s])
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start[s], ev_stop[s]));
            kernel_ms += ms;
            needs_timing[s] = false;
        }

        std::memcpy(h_pinned_target[s], target_seq.data(), target_len);

        CUDA_CHECK(cudaMemcpyAsync(d_target[s], h_pinned_target[s],
                                   target_len * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemsetAsync(d_max_score[s], 0, sizeof(int), streams[s]));
        CUDA_CHECK(cudaMemsetAsync(d_overflow_flag[s], 0, sizeof(int), streams[s]));

        int num_blocks_x = (query_len + TILE_SIZE - 1) / TILE_SIZE;
        int num_blocks_y = (target_len + TILE_SIZE - 1) / TILE_SIZE;
        int total_macro_diagonals = num_blocks_x + num_blocks_y - 1;

        CUDA_CHECK(cudaEventRecord(ev_start[s], streams[s]));

        for (int step = 0; step < total_macro_diagonals; step++)
        {
            int min_x = max(0, step - num_blocks_y + 1);
            int max_x = min(step, num_blocks_x - 1);
            int num_tiles = max_x - min_x + 1;

            align_massive_wavefront_tile_i16<<<num_tiles, TILE_SIZE, 0, streams[s]>>>(
                d_row_H[s], d_col_H[s], d_col_E[s], d_row_F[s], d_corner_H[s],
                d_query, d_target[s], query_len, target_len,
                step, num_blocks_x, num_blocks_y, algorithm,
                d_max_score[s], d_overflow_flag[s]);
        }

        CUDA_CHECK(cudaEventRecord(ev_stop[s], streams[s]));
        needs_timing[s] = true;

        if (algorithm == 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(&h_pinned_nw[i], &d_row_H[s][target_len - 1],
                                       sizeof(int16_t), cudaMemcpyDeviceToHost, streams[s]));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(&h_pinned_sw[i], d_max_score[s],
                                       sizeof(int), cudaMemcpyDeviceToHost, streams[s]));
        }
        CUDA_CHECK(cudaMemcpyAsync(&h_pinned_overflow[i], d_overflow_flag[s],
                                   sizeof(int), cudaMemcpyDeviceToHost, streams[s]));
    }

    // Drain all streams before reading the pinned result/flag buffers
    for (int s = 0; s < num_streams; s++)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        if (needs_timing[s])
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start[s], ev_stop[s]));
            kernel_ms += ms;
            needs_timing[s] = false;
        }
    }

    // Write int16 results — overflowed ones will be overwritten below
    for (size_t i = 0; i < num_large; i++)
    {
        if (algorithm == 0)
            results[result_offset + i] = h_pinned_nw[i];
        else
            results[result_offset + i] = h_pinned_sw[i];
    }

    // int32 fallback for any sequence whose int16 run overflowed. Rare — so
    // we serialize on the default stream with one shared buffer set.
    {
        bool any_overflow = false;
        for (size_t i = 0; i < num_large; i++)
        {
            if (h_pinned_overflow[i])
            {
                any_overflow = true;
                break;
            }
        }

        if (any_overflow)
        {
            CUDA_CHECK(cudaMalloc((void **)&d_row_H32, max_target_len * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **)&d_row_F32, max_target_len * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **)&d_col_H32, query_len * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **)&d_col_E32, query_len * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **)&d_corner_H32, max_num_tiles_total * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc((void **)&d_max_score32, sizeof(int)));
            CUDA_CHECK(cudaMalloc((void **)&d_target_fb, max_target_len * sizeof(unsigned char)));

            CUDA_CHECK(cudaEventCreate(&ev_fb_start));
            CUDA_CHECK(cudaEventCreate(&ev_fb_stop));

            for (size_t i = 0; i < num_large; i++)
            {
                if (!h_pinned_overflow[i])
                    continue;

                const auto &target_seq = large_seqs[i];
                const int target_len = target_seq.size();

                CUDA_CHECK(cudaMemcpy(d_target_fb, target_seq.data(),
                                      target_len * sizeof(unsigned char),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_max_score32, 0, sizeof(int)));

                int num_blocks_x = (query_len + TILE_SIZE - 1) / TILE_SIZE;
                int num_blocks_y = (target_len + TILE_SIZE - 1) / TILE_SIZE;
                int total_macro_diagonals = num_blocks_x + num_blocks_y - 1;

                CUDA_CHECK(cudaEventRecord(ev_fb_start));
                for (int step = 0; step < total_macro_diagonals; step++)
                {
                    int min_x = max(0, step - num_blocks_y + 1);
                    int max_x = min(step, num_blocks_x - 1);
                    int num_tiles = max_x - min_x + 1;

                    align_massive_wavefront_tile<<<num_tiles, TILE_SIZE>>>(
                        d_row_H32, d_col_H32, d_col_E32, d_row_F32, d_corner_H32,
                        d_query, d_target_fb, query_len, target_len,
                        step, num_blocks_x, num_blocks_y, algorithm, d_max_score32);
                }

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(ev_fb_stop));
                CUDA_CHECK(cudaDeviceSynchronize());
                {
                    float ms = 0.0f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_fb_start, ev_fb_stop));
                    kernel_ms += ms;
                }

                if (algorithm == 0)
                {
                    int32_t nw_score = 0;
                    CUDA_CHECK(cudaMemcpy(&nw_score, &d_row_H32[target_len - 1],
                                          sizeof(int32_t), cudaMemcpyDeviceToHost));
                    results[result_offset + i] = nw_score;
                }
                else
                {
                    int sw_score = 0;
                    CUDA_CHECK(cudaMemcpy(&sw_score, d_max_score32,
                                          sizeof(int), cudaMemcpyDeviceToHost));
                    results[result_offset + i] = sw_score;
                }
            }
        }
    }

    return_ms = kernel_ms;
    goto cleanup;

cleanup:
    if (ev_fb_start)
        cudaEventDestroy(ev_fb_start);
    if (ev_fb_stop)
        cudaEventDestroy(ev_fb_stop);
    cudaFree(d_query);
    for (int s = 0; s < MAX_STREAMS; s++)
    {
        if (ev_created[s])
        {
            cudaEventDestroy(ev_start[s]);
            cudaEventDestroy(ev_stop[s]);
        }
        cudaFree(d_target[s]);
        cudaFree(d_row_H[s]);
        cudaFree(d_row_F[s]);
        cudaFree(d_col_H[s]);
        cudaFree(d_col_E[s]);
        cudaFree(d_corner_H[s]);
        cudaFree(d_max_score[s]);
        cudaFree(d_overflow_flag[s]);
        if (h_pinned_target[s])
            cudaFreeHost(h_pinned_target[s]);
        if (stream_created[s])
            cudaStreamDestroy(streams[s]);
    }
    cudaFree(d_row_H32);
    cudaFree(d_row_F32);
    cudaFree(d_col_H32);
    cudaFree(d_col_E32);
    cudaFree(d_corner_H32);
    cudaFree(d_max_score32);
    cudaFree(d_target_fb);
    if (h_pinned_nw)
        cudaFreeHost(h_pinned_nw);
    if (h_pinned_sw)
        cudaFreeHost(h_pinned_sw);
    if (h_pinned_overflow)
        cudaFreeHost(h_pinned_overflow);

    return return_ms;
}

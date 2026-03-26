#include "intraAlignGPU.cuh"
#include "params.h"
#include "blosum62.h"
#include <cstdio>
#include <vector>
#define TILE_SIZE 32

__constant__ int8_t cuda_blosum62[24 * 24];

// Terminate if we get any errors from the device
#define CUDA_CHECK(call)                                                                                 \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t err = call;                                                                          \
        if (err != cudaSuccess)                                                                          \
        {                                                                                                \
            fprintf(stderr, "CUDA Error:\n  File: %s\n  Line: %d\n  Error code: %d\n  Error text: %s\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));                                   \
            return -1;                                                                                   \
        }                                                                                                \
    } while (0)

__global__ void
align_massive_wavefront_tile(
    int16_t *__restrict__ row_buf_H, int16_t *__restrict__ col_buf_H,
    int16_t *__restrict__ row_buf_E, int16_t *__restrict__ col_buf_E,
    int16_t *__restrict__ row_buf_F, int16_t *__restrict__ col_buf_F,
    int16_t *__restrict__ d_corner_H,
    const unsigned char *__restrict__ d_query, const unsigned char *__restrict__ d_target,
    const int query_len, const int target_len,
    const int step, const int num_blocks_x, const int num_blocks_y,
    const int algorithm, int *__restrict__ d_max_score)
{
    // Mapping: X is Query (Rows, sequence i), Y is Target (Cols, sequence j)
    int min_x = max(0, step - num_blocks_y + 1);
    int block_x = min_x + blockIdx.x; // Block row
    int block_y = step - block_x;     // Block col

    int tx = threadIdx.x; // Lane 0 to 31

    int global_i = block_x * TILE_SIZE + tx; // Query index
    int global_j = block_y * TILE_SIZE + tx; // Target index

    // Allocate 33x33 to hold the tile + top/left boundaries
    __shared__ int16_t H_tile[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ int16_t E_tile[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ int16_t F_tile[TILE_SIZE + 1][TILE_SIZE + 1];

    __shared__ unsigned char q_sub[TILE_SIZE];
    __shared__ unsigned char t_sub[TILE_SIZE];
    __shared__ int local_max; // Use int to prevent atomicMax overflow issues

    if (tx == 0)
        local_max = 0;

    // Load Subsequences
    if (global_i < query_len)
        q_sub[tx] = d_query[global_i];
    if (global_j < target_len)
        t_sub[tx] = d_target[global_j];

    // Load Boundaries
    // Top boundary (Indexed by Target/Cols)
    if (global_j < target_len)
    {
        if (block_x > 0)
        {
            H_tile[0][tx + 1] = row_buf_H[global_j];
            E_tile[0][tx + 1] = row_buf_E[global_j];
            F_tile[0][tx + 1] = row_buf_F[global_j];
        }
        else
        {
            H_tile[0][tx + 1] = (algorithm == 0) ? (-OPEN - (global_j)*EXTEND) : 0;
            E_tile[0][tx + 1] = -10000;
            F_tile[0][tx + 1] = -10000;
        }
    }

    // Left boundary (Indexed by Query/Rows)
    if (global_i < query_len)
    {
        if (block_y > 0)
        {
            H_tile[tx + 1][0] = col_buf_H[global_i];
            E_tile[tx + 1][0] = col_buf_E[global_i];
            F_tile[tx + 1][0] = col_buf_F[global_i];
        }
        else
        {
            H_tile[tx + 1][0] = (algorithm == 0) ? (-OPEN - (global_i)*EXTEND) : 0;
            E_tile[tx + 1][0] = -10000;
            F_tile[tx + 1][0] = -10000;
        }
    }

    // Top-left corner dependency for cell (1,1) of the tile
    if (tx == 0)
    {
        if (block_x > 0 && block_y > 0)
        {
            // Read directly from our dedicated corner grid
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

    __syncthreads(); // Ensure all boundaries and sequences are visible

    // The Wavefront Execution
    int num_diagonals = 2 * TILE_SIZE - 1;
    int thread_max = 0;

    for (int k = 0; k < num_diagonals; k++)
    {
        int j = tx + 1;
        int i = k - tx + 1; // Row calculation based on anti-diagonal step

        // Bounds check: inside tile AND inside actual sequence lengths
        if (i >= 1 && i <= TILE_SIZE)
        {
            int seq_i = block_x * TILE_SIZE + i - 1;
            int seq_j = block_y * TILE_SIZE + j - 1;

            if (seq_i < query_len && seq_j < target_len)
            {
                int16_t h_diag = H_tile[i - 1][j - 1];
                int16_t h_up = H_tile[i - 1][j];
                int16_t h_left = H_tile[i][j - 1];
                int16_t e_left = E_tile[i][j - 1];
                int16_t f_up = F_tile[i - 1][j];

                unsigned char q_char = q_sub[i - 1];
                unsigned char t_char = t_sub[j - 1];

                int16_t match_score = h_diag + cuda_blosum62[q_char * 24 + t_char];

                int16_t e_curr = max((int16_t)(h_left - OPEN), (int16_t)(e_left - EXTEND));
                int16_t f_curr = max((int16_t)(h_up - OPEN), (int16_t)(f_up - EXTEND));

                int16_t h_curr = max(match_score, max(e_curr, f_curr));

                if (algorithm == 1)
                { // SW specific floor and max tracking
                    h_curr = max((int16_t)0, h_curr);
                    if (h_curr > thread_max)
                        thread_max = h_curr;
                }

                H_tile[i][j] = h_curr;
                E_tile[i][j] = e_curr;
                F_tile[i][j] = f_curr;
            }
        }
        __syncthreads(); // sync warp before next diagonal
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

    // Write Boundaries Back to Global Buffers
    if (global_j < target_len)
    {
        row_buf_H[global_j] = H_tile[valid_i][tx + 1];
        row_buf_E[global_j] = E_tile[valid_i][tx + 1];
        row_buf_F[global_j] = F_tile[valid_i][tx + 1];
    }

    if (global_i < query_len)
    {
        col_buf_H[global_i] = H_tile[tx + 1][valid_j];
        col_buf_E[global_i] = E_tile[tx + 1][valid_j];
        col_buf_F[global_i] = F_tile[tx + 1][valid_j];
    }

    // SW Max Score Reduction
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

int intraAlignGPU(
    const int algorithm,
    const std::vector<unsigned char> &query_seq,
    const std::vector<unsigned char> &target_seq,
    unsigned char *d_query, unsigned char *d_target,
    int16_t *d_row_H, int16_t *d_row_E, int16_t *d_row_F,
    int16_t *d_col_H, int16_t *d_col_E, int16_t *d_col_F,
    int16_t *d_corner_H, int *d_max_score)
{
    const int query_len = query_seq.size();
    const int target_len = target_seq.size();

    int num_blocks_x = (query_len + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks_y = (target_len + TILE_SIZE - 1) / TILE_SIZE;
    int total_macro_diagonals = num_blocks_x + num_blocks_y - 1;

    int host_max_score = 0;
    int final_score = 0;

    // Reset SW max score for this specific run
    CUDA_CHECK(cudaMemcpy(d_max_score, &host_max_score, sizeof(int), cudaMemcpyHostToDevice));

    // ONLY copy the target sequence (Query is copied once in main)
    CUDA_CHECK(cudaMemcpy(d_target, target_seq.data(), target_len * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Load BLOSUM matrix
    CUDA_CHECK(cudaMemcpyToSymbol(cuda_blosum62, blosum62, sizeof(int8_t) * 24 * 24));

    // Execute Macro-Wavefront Loop
    for (int step = 0; step < total_macro_diagonals; step++)
    {
        int min_x = max(0, step - num_blocks_y + 1);
        int max_x = min(step, num_blocks_x - 1);
        int num_tiles = max_x - min_x + 1;

        align_massive_wavefront_tile<<<num_tiles, TILE_SIZE>>>(
            d_row_H, d_col_H, d_row_E, d_col_E, d_row_F, d_col_F, d_corner_H,
            d_query, d_target, query_len, target_len,
            step, num_blocks_x, num_blocks_y, algorithm, d_max_score);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch Final Score
    if (algorithm == 0)
    {
        int16_t nw_score;
        CUDA_CHECK(cudaMemcpy(&nw_score, &d_row_H[target_len - 1], sizeof(int16_t), cudaMemcpyDeviceToHost));
        final_score = nw_score;
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(&final_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost));
    }

    return final_score;
}
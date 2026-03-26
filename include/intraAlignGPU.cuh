#pragma once

#include <vector>

int intraAlignGPU(
    const int algorithm,
    const std::vector<unsigned char> &query_seq,
    const std::vector<unsigned char> &target_seq,
    unsigned char *d_query, unsigned char *d_target,
    int16_t *d_row_H, int16_t *d_row_E, int16_t *d_row_F,
    int16_t *d_col_H, int16_t *d_col_E, int16_t *d_col_F,
    int16_t *d_corner_H, int *d_max_score);
#pragma once

#include <vector>

int intraAlignGPU(
    int algorithm,
    const std::vector<unsigned char> &query_seq,
    const std::vector<std::vector<unsigned char>> &massive_sequences,
    std::vector<int> &results,
    int result_offset);
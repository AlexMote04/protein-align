#pragma once

#include <vector>

int naiveInterGPU(int algorithm, std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets);
#pragma once

#include "blosum62.h"
#include "params.h"
#include <vector>
#include <cstdint>
#include <algorithm>

int align_sw(const unsigned char *query_seq, const unsigned char *db_residues, const int M, const int N);
int swCPU(std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets);
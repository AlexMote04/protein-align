#pragma once

#include "blosum62.h"
#include "params.h"
#include <vector>

int align_sw(const unsigned char *query_seq, const unsigned char *target_seq, const int M, const int N, std::vector<int16_t> &H, std::vector<int16_t> &E, std::vector<int16_t> &F);
int align_nw(const unsigned char *query_seq, const unsigned char *target_seq, const int M, const int N, std::vector<int16_t> &H, std::vector<int16_t> &E, std::vector<int16_t> &F);
int alignCPU(const int algorithm, std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets);
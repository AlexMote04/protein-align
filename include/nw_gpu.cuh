#pragma once

#include "params.h"
#include "blosum62.h"
#include <vector>
#include <cstdint>

int nwGPU(std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets);
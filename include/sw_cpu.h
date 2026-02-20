#pragma once

#include "blosum62.h"
#include "params.h"
#include <vector>
#include <cstdint>
#include <algorithm>

int swCPU(std::vector<int> &scores, const std::vector<unsigned char> &query, const std::vector<unsigned char> &db, const std::vector<int> &offsets);
int align(const std::vector<unsigned char> &query, const std::vector<unsigned char> &db, int slen);

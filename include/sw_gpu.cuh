#pragma once

#include "params.h"
#include "blosum62.h"
#include <vector>
#include <cstdint>

int swGPU(std::vector<int> &scores, const std::vector<unsigned char> &query, const std::vector<unsigned char> &db, const std::vector<int> &offsets);
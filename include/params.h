// Same params as PSISEARCH
// BLOSUM62, affine gap open (10), gap extend (1)
#pragma once

#include <cstdint>

#define TILE_SIZE 32

constexpr int16_t OPEN = 10;
constexpr int16_t EXTEND = 1;
constexpr int MAX_NUM_SEQS = 100000;
constexpr int MAX_QUERY_LEN = 8192;
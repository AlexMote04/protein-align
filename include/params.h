// Same params as PSISEARCH
// BLOSUM62, affine gap open (10), gap extend (1)
#pragma once

#include <cstdint>

constexpr int32_t OPEN = 10;
constexpr int32_t EXTEND = 1;
constexpr int MAX_QUERY_LEN = 8192;
#define NEG_INF_32 (INT32_MIN / 2)

// Flag any cell whose score reaches this magnitude — leaves margin for one
// more OPEN/EXTEND subtraction before int16 arithmetic would actually wrap.
constexpr int16_t OVERFLOW_THRESHOLD_16 = 30000;
constexpr int16_t NEG_INF_16 = -30000;
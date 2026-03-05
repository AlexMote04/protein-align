#include "parasail.h"
#include "parasail/matrices/blosum62.h"
#include "params.h"
#include <vector>
#include <string>

void nwParasail(std::vector<int> &cpu_scores, const std::string &query, const std::vector<std::string> &db_ascii);
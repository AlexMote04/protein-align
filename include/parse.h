#pragma once

#include <string>
#include <vector>

int parseQuery(std::vector<int> query, std::string query_path);
int parseDB(std::vector<int> db, std::vector<int> offsets, std::string db_path, int db_size);
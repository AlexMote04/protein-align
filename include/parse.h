#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

const int MAX_DB_LENGTH = 2977;

void initConversionTable();
int parseQuery(std::vector<unsigned char> &query, std::string query_path);
int parseDB(std::vector<unsigned char> &db, std::vector<int> &offsets, std::string db_path, int db_size);
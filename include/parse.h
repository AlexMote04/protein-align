#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

unsigned char amino_to_uchar[128];

void initConversionTable();
int parseQuery(std::vector<unsigned char> query, std::string query_path);
int parseDB(std::vector<unsigned char> db, std::vector<int> offsets, std::string db_path, int db_size);
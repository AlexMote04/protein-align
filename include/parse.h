#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

void initConversionTable();
void parseQuery(std::vector<unsigned char> &query_seq, const std::string &query_path);
void parseQueryParasail(std::string &query_seq, const std::string &query_path);
std::vector<std::string> loadOrCacheDatabase(const std::string &fasta_path, int num_seqs);
void generateDBSoA(const std::vector<std::string> &sorted_db,
                   std::vector<unsigned char> &db_residues,
                   std::vector<int> &db_offsets,
                   std::vector<std::vector<unsigned char>> &massive_sequences,
                   int threshold);
void generateDBParasail(const std::vector<std::string> &sorted_db, std::vector<std::string> &db_ascii);
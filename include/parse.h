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
void parseDB(std::vector<unsigned char> &db_residues, std::vector<int> &db_offsets, const std::string &db_path, int num_seqs);
void parseDBSoA(std::vector<unsigned char> &db_residues, std::vector<int> &db_offsets, const std::string &db_path, int num_seqs);
void parseDBParasail(std::vector<std::string> &db_ascii, const std::string &db_path, int num_seqs);
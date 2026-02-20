#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>

#include "parse.h"
#include "sw_cpu.h"

const std::string QUERY_PATH = "data/input/query.fasta";
const std::string DB_PATH = "data/input/db.fasta";
const std::string SCORES_PATH = "data/output/scores.txt";

int main(int argc, char** argv){
  if(argc != 3){
    std::cerr << "Usage: ./bin/align <mode (cpu, gpu)> <db_size>" << std::endl;
    return 1;
  }

  std::string mode = argv[1];
  int db_size = std::atoi(argv[2]);

  if (db_size < 1 || db_size > 2977){
    std::cerr << "Error: db_size must be between 1 and 2977" << std::endl;
  }

  std::vector<unsigned char> query;
  std::vector<unsigned char> db;
  std::vector<int> offsets;
  initConversionTable();

  if(parseQuery(query, QUERY_PATH) == -1){
    return 1;
  };
  if(parseDB(db, offsets, DB_PATH, db_size) == -1){
    return 1;
  };

  std::vector<int> scores(offsets.size()); // Pre-allocate scores vector

  if (mode == "cpu"){
    swCPU(scores, query, db, offsets);
  }
  else if (mode == "gpu"){
    swGPU(scores, query, db, offsets);
  } else {
    std::cerr << "Error: Invalid Mode" << std::endl;
    return 1;
  }

  // Sort scores by score

  // Write alignment scores to output file
  std::ofstream scores_file(SCORES_PATH);
  scores_file << "scores" << std::endl;

  for (auto score: scores){
    scores_file << scores << std::endl;
  }
  scores_file.close();
  return 0;
}
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>

#include "parse.h"

const std::string QUERY_PATH = "data/input/query.fasta";
const std::string DB_PATH = "data/input/db.fasta";
const std::string SCORES_PATH = "data/output/scores.txt";

int main(int argc, char** argv){
  if(argc != 3){
    std::cerr << "Usage: ./bin/main <mode (cpu, gpu)> <db_size>" << std::endl;
    return 1;
  }

  std::string mode = argv[1];
  int db_size = std::atoi(argv[2]);

  if (db_size < 1 || db_size > 500000){
    std::cerr << "Error: db_size must be between 1 and 500,000" << std::endl;
  }

  std::vector<unsigned char> query;
  std::vector<unsigned char> db;
  std::vector<int> offsets;
  std::vector<int> scores;
  initConversionTable();

  if(parseQuery(query, QUERY_PATH) == -1){
    std::cerr << "Error: Failed to parse query file. Ensure query file is in the correct format and that the query path is valid" << std::endl;
  };
  if(parseDB(db, offsets, DB_PATH, db_size) == -1){
    std::cerr << "Error: Failed to parse database file. Ensure database file is in the correct format and that the database path is valid" << std::endl;
  };

  if (mode == "cpu"){
    swCPU(scores, query, db);
  }
  else if (mode == "gpu"){
    swGPU(scores, query, db);
  } else {
    std::cerr << "Error: Invalid Mode" << std::endl;
    return 1;
  }

  // Write alignment scores to output file
  std::ofstream scores_file(SCORES_PATH);
  scores_file << "scores" << std::endl;

  for (auto score: scores){
    scores_file << scores << std::endl;
  }
  scores_file.close();
  return 0;
}
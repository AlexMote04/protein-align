#include "catch.hpp"
#include "parse.h"
#include "sw_cpu.h"
#include "vector"

TEST_CASE("smoke test")
{
  std::vector<unsigned char> query_seq{4};               // C
  std::vector<unsigned char> db_residues{0, 1, 2, 3, 4}; // A, R, N, D, C
  std::vector<int> db_offsets{0, 1, 2, 3, 4, 5};
  std::vector<int> scores(5);
  std::vector<int> solution{0, 0, 0, 0, 9};

  swCPU(scores, query_seq, db_residues, db_offsets);

  CHECK(scores == solution);
}

TEST_CASE("sw correctness")
{
  std::string BASE_PATH = "../data/test/align/";
  std::string DATABASE_PATH = BASE_PATH + "db";

  SECTION("test0")
  {
    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, BASE_PATH + "q0");
    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    parseDB(db_residues, db_offsets, DATABASE_PATH, 12);
    std::vector<int> scores(12);
    std::vector<int> solution{17, 12, 20, 19, 17, 19, 17, 17, 22, 14, 19, 16};
    swCPU(scores, query_seq, db_residues, db_offsets);
    CHECK(scores == solution);
  }

  SECTION("test1")
  {
    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, BASE_PATH + "q1");
    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    parseDB(db_residues, db_offsets, DATABASE_PATH, 12);
    std::vector<int> scores(12);
    std::vector<int> solution{17, 14, 21, 16, 19, 20, 18, 17, 12, 11, 23, 20};
    swCPU(scores, query_seq, db_residues, db_offsets);
    CHECK(scores == solution);
  }
}
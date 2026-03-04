#include "catch.hpp"
#include "parse.h"
#include "nw_cpu.h"
#include "vector"

TEST_CASE("nw correctness")
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
    std::vector<int> solution{-74, -69, -46, -33, -42, -42, -34, -47, -64, -77, -58, -59};
    nwCPU(scores, query_seq, db_residues, db_offsets);
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
    std::vector<int> solution{-78, -62, -53, -37, -39, -41, -43, -52, -54, -73, -43, -42};
    nwCPU(scores, query_seq, db_residues, db_offsets);
    CHECK(scores == solution);
  }
}
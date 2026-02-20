#include "catch.hpp"
#include "parse.h"
#include "sw_gpu.cuh"
#include "vector"

TEST_CASE("gpu smoke test"){
  std::vector<unsigned char> query{4}; // C
  std::vector<unsigned char> db{0, 1, 2, 3, 4}; // A, R, N, D, C
  std::vector<int> offsets{0, 1, 2, 3, 4};
  std::vector<int> scores(5);
  std::vector<int> solution{0, 0, 0, 0, 9};

  swGPU(scores, query, db, offsets);

  CHECK(scores == solution);
}

TEST_CASE("gpu correctness"){
  std::string BASE_PATH = "../data/test/auto/";

  SECTION("test0"){
    std::vector<unsigned char> query;
    parseQuery(query, BASE_PATH + "q0");
    std::vector<unsigned char> db;
    std::vector<int> offsets;
    parseDB(db, offsets, BASE_PATH + "db0", 12);
    std::vector<int> scores(12);
    std::vector<int> solution{17, 12, 20, 19, 17, 19, 17, 17, 22, 14, 19, 16};
    swGPU(scores, query, db, offsets);
    CHECK(scores == solution);
  }

  SECTION("test1"){
    std::vector<unsigned char> query;
    parseQuery(query, BASE_PATH + "q1");
    std::vector<unsigned char> db;
    std::vector<int> offsets;
    parseDB(db, offsets, BASE_PATH + "db1", 12);
    std::vector<int> scores(12);
    std::vector<int> solution{20, 32, 23, 15, 18, 23, 18, 19, 12, 18, 19, 15};
    swGPU(scores, query, db, offsets);
    CHECK(scores == solution);
  }
}
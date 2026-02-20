#include "catch.hpp"
#include "parse.h"

std::string BASE_DIR = "../data/test/manual/";

// Query Parser
TEST_CASE("basic query parser functionality", "[query parser]"){
  SECTION("invalid file path"){
    std::vector<unsigned char> query;
    REQUIRE_THROWS_WITH(parseQuery(query, "invalid/path"), "Could not open file: invalid/path");
  }

  SECTION("empty query"){
    std::vector<unsigned char> query;
    REQUIRE_THROWS_WITH(parseQuery(query, BASE_DIR + "empty"), "Invalid query file: No query sequence found");
  }

  SECTION("multiline char sequence"){
    std::vector<unsigned char> solution = {0, 1, 2, 3};
    std::vector<unsigned char> query;
    parseQuery(query, BASE_DIR + "query_multiline");
    CHECK(query == solution);
  }
}

TEST_CASE("query parsing correctness", "[query parser]") {
  std::vector<unsigned char> solution;
  for(int i = 0; i < 24; i++){
    solution.push_back(i); // One for each valid code
  }

  solution.push_back(22); // Invalid character

  SECTION("upper case characters are parsed correctly") {
    std::vector<unsigned char> query;
    parseQuery(query, BASE_DIR + "query_all_upper"); // ARN....

    CHECK(query == solution);
  }

  SECTION("lower case characters are parsed correctly") {
    std::vector<unsigned char> query;
    parseQuery(query, BASE_DIR + "query_all_lower"); // arn....

    CHECK(query == solution);
  }
}

// Database Parser
TEST_CASE("basic database parser functionality", "[database parser]"){
  SECTION("invalid file path"){
    std::vector<unsigned char> db;
    std::vector<int> offsets;
    int db_size = 10;
    REQUIRE_THROWS_WITH(parseDB(db, offsets, "invalid/path", db_size), "Could not open file: invalid/path");
  }

  SECTION("empty database"){
    std::vector<unsigned char> db;
    std::vector<int> offsets;
    int db_size = 10;
    REQUIRE_THROWS_WITH(parseDB(db, offsets, BASE_DIR + "empty", db_size), "Invalid database file: No sequences found");
  }

  SECTION("db_size greater than number of sequences in database"){
    std::vector<unsigned char> db;
    std::vector<int> offsets;
    int db_size = 4;
    REQUIRE_THROWS_WITH(parseDB(db, offsets, BASE_DIR + "db_small", db_size), "Tried parsing 4 sequences but only found 3 in database file");
  }
}

TEST_CASE("database parsing correctness", "[database parser]"){
  SECTION("small db correctly parsed"){
    std::vector<unsigned char> db_correct;
    std::vector<int> offsets_correct {0, 23, 46};

    for(int i = 0; i < 23; i++){
      db_correct.push_back(i); // Lowercase characters
    }
    for(int i = 0; i < 23; i++){
      db_correct.push_back(i); // Uppercase characters
    }
    for(int i = 0; i < 24; i++){
      db_correct.push_back(i); // Mixed characters and '*'
    }
    db_correct.push_back(22); // Invalid character

    std::vector<unsigned char> db;
    std::vector<int> offsets;

    int db_size = 3;
    int status = parseDB(db, offsets, BASE_DIR + "db_small", db_size);
    CHECK(status == 0);
    CHECK(db == db_correct);
    CHECK(offsets == offsets_correct);
  }

  SECTION("db_size less than number of sequences in database"){
    std::vector<unsigned char> db_correct;
    std::vector<int> offsets_correct {0, 23};

    for(int i = 0; i < 23; i++){
      db_correct.push_back(i); // Lowercase characters
    }
    for(int i = 0; i < 23; i++){
      db_correct.push_back(i); // Uppercase characters
    }

    std::vector<unsigned char> db;
    std::vector<int> offsets;

    int db_size = 2;
    int status = parseDB(db, offsets, BASE_DIR + "db_small", db_size);
    CHECK(status == 0);
    CHECK(db == db_correct);
    CHECK(offsets == offsets_correct);
  }

  SECTION("parse maximum database length"){
    std::vector<unsigned char> db;
    std::vector<int> offsets;
    int status = parseDB(db, offsets, "../data/input/uniprot_sprot", MAX_DB_LENGTH);
    CHECK(status == 0);
    CHECK(offsets.size() == MAX_DB_LENGTH);
  }
}
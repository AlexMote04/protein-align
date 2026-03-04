#include "catch.hpp"
#include "parse.h"

std::string BASE_DIR = "../data/test/parser/";

// Query Parser
TEST_CASE("basic query parser functionality", "[query parser]")
{
  SECTION("invalid file path")
  {
    std::vector<unsigned char> query_seq;
    REQUIRE_THROWS_WITH(parseQuery(query_seq, "invalid/path"), "Could not open file: invalid/path");
  }

  SECTION("empty query")
  {
    std::vector<unsigned char> query_seq;
    REQUIRE_THROWS_WITH(parseQuery(query_seq, BASE_DIR + "empty"), "Invalid query file: No query sequence found");
  }

  SECTION("multiline char sequence")
  {
    std::vector<unsigned char> solution = {0, 1, 2, 3};
    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, BASE_DIR + "query_multiline");
    CHECK(query_seq == solution);
  }
}

TEST_CASE("query parsing correctness", "[query parser]")
{
  std::vector<unsigned char> solution;
  for (int i = 0; i < 24; i++)
  {
    solution.push_back(i); // One for each valid code
  }

  solution.push_back(22); // Invalid character

  SECTION("upper case characters are parsed correctly")
  {
    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, BASE_DIR + "query_all_upper"); // ARN....

    CHECK(query_seq == solution);
  }

  SECTION("lower case characters are parsed correctly")
  {
    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, BASE_DIR + "query_all_lower"); // arn....

    CHECK(query_seq == solution);
  }
}

// Database Parser
TEST_CASE("basic database parser functionality", "[database parser]")
{
  SECTION("invalid file path")
  {
    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    int num_sequences = 10;
    REQUIRE_THROWS_WITH(parseDB(db_residues, db_offsets, "invalid/path", num_sequences), "Could not open file: invalid/path");
  }

  SECTION("empty database")
  {
    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    int num_sequences = 10;
    REQUIRE_THROWS_WITH(parseDB(db_residues, db_offsets, BASE_DIR + "empty", num_sequences), "Invalid database file: No sequences found");
  }

  SECTION("num_sequences greater than number of sequences in database")
  {
    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    int num_sequences = 4;
    REQUIRE_THROWS_WITH(parseDB(db_residues, db_offsets, BASE_DIR + "db_small", num_sequences), "Tried parsing 4 sequences but only found 3 in database file");
  }
}

TEST_CASE("database parsing correctness", "[database parser]")
{
  SECTION("small db correctly parsed")
  {
    std::vector<unsigned char> db_correct;
    std::vector<int> offsets_correct{0, 23, 46, 71};

    for (int i = 0; i < 23; i++)
    {
      db_correct.push_back(i); // Lowercase characters
    }
    for (int i = 0; i < 23; i++)
    {
      db_correct.push_back(i); // Uppercase characters
    }
    for (int i = 0; i < 24; i++)
    {
      db_correct.push_back(i); // Mixed characters and '*'
    }
    db_correct.push_back(22); // Invalid character

    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;

    int num_sequences = 3;
    parseDB(db_residues, db_offsets, BASE_DIR + "db_small", num_sequences);
    CHECK(db_residues == db_correct);
    CHECK(db_offsets == offsets_correct);
  }

  SECTION("num_sequences less than number of sequences in database")
  {
    std::vector<unsigned char> db_correct;
    std::vector<int> offsets_correct{0, 23, 46};

    for (int i = 0; i < 23; i++)
    {
      db_correct.push_back(i); // Lowercase characters
    }
    for (int i = 0; i < 23; i++)
    {
      db_correct.push_back(i); // Uppercase characters
    }

    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;

    int num_sequences = 2;
    parseDB(db_residues, db_offsets, BASE_DIR + "db_small", num_sequences);
    CHECK(db_residues == db_correct);
    CHECK(db_offsets == offsets_correct);
  }

  SECTION("parse maximum database length")
  {
    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    parseDB(db_residues, db_offsets, "../data/input/uniprot_sprot.fasta", MAX_NUM_SEQS);
    CHECK((db_offsets.size() - 1) == MAX_NUM_SEQS);
  }
}
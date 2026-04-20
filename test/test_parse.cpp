#include "catch.hpp"
#include "parse.h"
#include "params.h"
#include <vector>
#include <string>
#include <fstream>

const std::string BASE_DIR = "../data/test/parser/";

// Helper to write temporary FASTA files for testing I/O
void createTempFasta(const std::string &path, const std::string &content)
{
    std::ofstream out(path);
    out << content;
    out.close();
}

// ==================================================================
// QUERY PARSING & ENCODING
// ==================================================================

TEST_CASE("Query parser encoding and functionality", "[query_parser]")
{
    SECTION("Standard amino acids encoded correctly")
    {
        createTempFasta(BASE_DIR + "temp_query.fasta", ">header\nARNDCQEGHILKMFPSTWYV");
        std::vector<unsigned char> query_seq;
        parseQuery(query_seq, BASE_DIR + "temp_query.fasta");

        // Expected mapping based on initConversionTable
        std::vector<unsigned char> expected = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
        CHECK(query_seq == expected);
    }

    SECTION("Unknown and special characters map to 23")
    {
        createTempFasta(BASE_DIR + "temp_unknown.fasta", ">header\nJOU*?");
        std::vector<unsigned char> query_seq;
        parseQuery(query_seq, BASE_DIR + "temp_unknown.fasta");

        // J, O, U, *, and ? should all map to 23
        std::vector<unsigned char> expected = {23, 23, 23, 23, 23};
        CHECK(query_seq == expected);
    }

    SECTION("Parasail parser retains pure ASCII")
    {
        createTempFasta(BASE_DIR + "temp_parasail.fasta", ">header\nARND");
        std::string query_ascii;
        parseQueryParasail(query_ascii, BASE_DIR + "temp_parasail.fasta");

        CHECK(query_ascii == "ARND");
    }
}

// ==================================================================
// SOA DATABASE GENERATION (In-Memory Logic)
// ==================================================================

TEST_CASE("generateDBSoA correctly pads and splits sequences", "[soa_generator]")
{
    // We bypass loadOrCacheDatabase to test the pure data-transformation logic
    // Sequences must be sorted by length (as guaranteed by loadOrCacheDatabase)
    std::vector<std::string> sorted_db = {
        "A",   // Length 1
        "CD",  // Length 2
        "EFG", // Length 3
        "HILK" // Length 4
    };

    std::vector<unsigned char> db_residues;
    std::vector<int> db_offsets;
    std::vector<std::vector<unsigned char>> massive_sequences;

    SECTION("Splits massive sequences correctly based on threshold")
    {
        int threshold = 3; // "EFG" and "HILK" should become massive
        generateDBGPU(sorted_db, db_residues, db_offsets, massive_sequences, threshold);

        // Check Split
        CHECK(db_offsets.size() == 3); // 2 small seqs + 1 tail offset
        CHECK(massive_sequences.size() == 2);

        // Check Massive Encoding
        std::vector<unsigned char> expected_massive_1 = {6, 13, 7};     // E, F, G
        std::vector<unsigned char> expected_massive_2 = {8, 9, 10, 11}; // H, I, L, K
        CHECK(massive_sequences[0] == expected_massive_1);
        CHECK(massive_sequences[1] == expected_massive_2);
    }

    SECTION("SoA padding logic and batch chunking")
    {
        int threshold = 10; // All sequences are "small"
        generateDBGPU(sorted_db, db_residues, db_offsets, massive_sequences, threshold);

        // Total 4 sequences. Fits in exactly 1 batch of 32.
        // Max length in batch = 4 ("HILK").
        // Total residues allocated should be 4 (len) * 32 (batch size) = 128.
        REQUIRE(db_residues.size() == 128);

        // Check Character 0 (Across all 32 threads)
        CHECK(db_residues[0] == 0);  // 'A'
        CHECK(db_residues[1] == 4);  // 'C'
        CHECK(db_residues[2] == 6);  // 'E'
        CHECK(db_residues[3] == 8);  // 'H'
        CHECK(db_residues[4] == 23); // Padding for empty slot

        // Check Character 1
        CHECK(db_residues[32] == 23); // 'A' has ended, so padding
        CHECK(db_residues[33] == 3);  // 'D'
        CHECK(db_residues[34] == 13); // 'F'
        CHECK(db_residues[35] == 9);  // 'I'
        CHECK(db_residues[36] == 23); // Padding
    }

    SECTION("db_offsets calculates unpadded start positions")
    {
        int threshold = 10;
        generateDBGPU(sorted_db, db_residues, db_offsets, massive_sequences, threshold);

        // Lengths: 1, 2, 3, 4
        std::vector<int> expected_offsets = {
            0, // Start of "A"
            1, // Start of "CD"
            3, // Start of "EFG"
            6, // Start of "HILK"
            10 // Tail offset (total unpadded characters)
        };

        CHECK(db_offsets == expected_offsets);
    }
}
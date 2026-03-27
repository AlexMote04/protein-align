#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#include "parse.h"
#include "alignCPU.h"
#include "interAlignGPU.cuh"
#include "intraAlignGPU.cuh"
#include "alignParasail.h"

const std::string QUERY_PATH = "../data/input/query";
const std::string DB_PATH = "../data/input/uniprot_sprot.fasta";

// This program calculates the alignment scores for a query sequence against a database of target sequences using multiple different
// Implementations of NW and SW and measure their performance.
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: ./bin/benchmark <algorithm> <num_seqs> <threshold>\n";
        return 1;
    }

    int algorithm{};
    if (std::string(argv[1]) == "nw")
        algorithm = 0;
    else if (std::string(argv[1]) == "sw")
        algorithm = 1;
    else
    {
        std::cerr << "Error: Unknown Algorithm. Only sw and nw are supported." << std::endl;
        return 1;
    }

    int num_seqs{};
    int threshold{};

    try
    {
        num_seqs = std::stoi(argv[2]);
        threshold = std::stoi(argv[3]);

        if (num_seqs <= 0 || threshold <= 0)
            throw std::invalid_argument("Values must be positive integers.");
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid number format. " << e.what() << "\n";
        std::cerr << "Usage: ./bin/benchmark <algorithm> <num_seqs> <threshold>\n";
        return 1;
    }
    catch (const std::out_of_range &e)
    {
        std::cerr << "Error: Number provided is out of integer range.\n";
        return 1;
    }

    initConversionTable();

    // Data for Custom Implementation
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> db_residues_soa;
    std::vector<int> db_offsets;
    std::vector<std::vector<unsigned char>> massive_sequences;

    // Data for Parasail (ASCII)
    std::string query_ascii;
    std::vector<std::string> db_ascii;

    try
    {
        // Load Queries
        parseQuery(query_seq, QUERY_PATH);
        parseQueryParasail(query_ascii, QUERY_PATH);

        // Load or Cache the Sorted Database
        std::vector<std::string> sorted_db = loadOrCacheDatabase(DB_PATH, num_seqs);

        // Generate Memory Layouts
        generateDBSoA(sorted_db, db_residues_soa, db_offsets, massive_sequences, threshold);
        generateDBParasail(sorted_db, db_ascii);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Parsing Error: " << e.what() << '\n';
        return 1;
    }

    if (query_seq.size() > MAX_QUERY_LEN)
    {
        fprintf(stderr, "Error: Query length exceeds MAX_QUERY_LEN.\n");
        return -1;
    }

    // ==========================================
    // WARM-UP PHASE
    // ==========================================

    // Create a tiny dummy dataset
    std::vector<unsigned char> dummy_query = {0, 1, 2, 3}; // "ARND"
    std::vector<std::string> dummy_db = {"ARND"};
    std::vector<unsigned char> dummy_residues_soa{};
    std::vector<int> dummy_offsets{};
    std::vector<std::vector<unsigned char>> dummy_massive_sequences{};
    generateDBSoA(dummy_db, dummy_residues_soa, dummy_offsets, dummy_massive_sequences, 1000);
    std::vector<int> dummy_scores(1);

    interAlignGPU(algorithm, dummy_scores, dummy_query, dummy_residues_soa, dummy_offsets);

    std::vector<int> parasail_scores(num_seqs);
    std::vector<int> gpu_scores(num_seqs);

    // ==========================================
    // TIMING PHASE
    // ==========================================

    auto parasail_start = std::chrono::high_resolution_clock::now();
    alignParasail(algorithm, parasail_scores, query_ascii, db_ascii);
    auto parasail_end = std::chrono::high_resolution_clock::now();

    size_t num_small_seqs = num_seqs - massive_sequences.size();
    size_t num_big_seqs = massive_sequences.size();

    auto gpu_inter_start = std::chrono::high_resolution_clock::now();
    if (num_small_seqs > 0)
        interAlignGPU(algorithm, gpu_scores, query_seq, db_residues_soa, db_offsets);
    auto gpu_inter_end = std::chrono::high_resolution_clock::now();

    auto gpu_intra_start = std::chrono::high_resolution_clock::now();
    if (num_big_seqs > 0)
        intraAlignGPU(algorithm, query_seq, massive_sequences, gpu_scores, num_small_seqs);
    auto gpu_intra_end = std::chrono::high_resolution_clock::now();

    // ==========================================
    // VERIFICATION PHASE
    // ==========================================

    int gpu_mismatches = 0;
    for (size_t i = 0; i < num_seqs; i++)
    {
        if (gpu_scores[i] != parasail_scores[i])
        {
            if (gpu_mismatches < 5)
            { // Only print the first 5 to avoid terminal spam
                std::cerr << "--- MISMATCH AT SEQUENCE " << i << " ---\n";
                std::cerr << "Target String  : " << db_ascii[i] << "\n";
                std::cerr << "Target Length  : " << db_ascii[i].length() << " residues\n";
                std::cerr << "GPU Score: " << gpu_scores[i] << "\n";
                std::cerr << "Parasail Score : " << parasail_scores[i] << "\n";
                std::cerr << "Difference     : " << std::abs(gpu_scores[i] - parasail_scores[i]) << "\n\n";
            }
            gpu_mismatches++;
        }
    }

    if (gpu_mismatches > 0)
    {
        std::cerr << "FAILED: " << gpu_mismatches << " GPU scores did not match Parasail!\n";
        return 1;
    }

    // ==========================================
    // METRICS & REPORTING (Console)
    // ==========================================
    std::chrono::duration<double, std::milli> parasail_ms = parasail_end - parasail_start;
    std::chrono::duration<double, std::milli> gpu_inter_ms = gpu_inter_end - gpu_inter_start;
    std::chrono::duration<double, std::milli> gpu_intra_ms = gpu_intra_end - gpu_intra_start;
    std::chrono::duration<double, std::milli> gpu_total_ms = gpu_inter_ms + gpu_intra_ms;

    size_t total_db_residues = 0;
    for (const auto &seq : db_ascii)
    {
        total_db_residues += seq.length();
    }
    double total_cell_updates = static_cast<double>(query_ascii.size()) * total_db_residues;
    double parasail_gcups = total_cell_updates / (parasail_ms.count() * 1e6);
    double gpu_gcups = total_cell_updates / (gpu_total_ms.count() * 1e6);

    // Latency: PARASAIL, OPTIMISED GPU. THROUGHPUT PARASAIL, OPTIMISED GPU. SPEEDUP: Parasail vs. Optimised
    printf("%.1f, %.1f, ", parasail_ms.count(), gpu_total_ms.count());
    printf("%.2f, %.2f, ", parasail_gcups, gpu_gcups);
    printf("%.1f\n", parasail_ms.count() / gpu_total_ms.count());

    return 0;
}
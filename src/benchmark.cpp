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
#include "naiveInterGPU.cuh"
#include "alignParasail.h"

#define num_repeats 3

// This warms up the GPU for before benchmarking
void warmUpGpu()
{
    std::vector<unsigned char> dummy_query = {0, 1, 2, 3}; // "ARND"
    std::vector<std::string> dummy_db = {"ARND"};
    std::vector<unsigned char> dummy_residues_soa{};
    std::vector<int> dummy_offsets{};
    std::vector<std::vector<unsigned char>> dummy_massive_sequences{};
    generateDBGPU(dummy_db, dummy_residues_soa, dummy_offsets, dummy_massive_sequences, 100000);
    std::vector<int> dummy_scores(1);

    interAlignGPU(0, dummy_scores, dummy_query, dummy_residues_soa, dummy_offsets);
}

// This verifies that our implementation scores match parasail
void checkResults(const std::vector<int> &correct_scores, const std::vector<int> &test_scores, std::string version_name)
{
    int mismatches = 0;
    for (int i = 0; i < correct_scores.size(); i++)
    {
        if (correct_scores[i] != test_scores[i])
        {
            printf("Mismatch at index %d!\n", i);
            printf("Correct score: %d\nTest score: %d\n", correct_scores[i], test_scores[i]);
            mismatches++;
        }
        if (mismatches > 2)
            break;
    }

    if (mismatches > 0)
        throw std::runtime_error("FAILED: " + version_name + " scores did not match Parasail!\n");
}

// This runs parasail n times and returns the average run time
std::chrono::duration<double, std::milli> runParasail(int algorithm, const std::string &query_path, const std::vector<std::string> &sorted_db, std::vector<int> &parasail_scores)
{
    // This will provide the gold standard used for verification
    std::string query_ascii;
    std::vector<std::string> db_ascii;
    parseQueryParasail(query_ascii, query_path);
    generateDBParasail(sorted_db, db_ascii);

    std::chrono::duration<double, std::milli> total_ms{};

    for (int i = 0; i < num_repeats; i++)
    {
        auto parasail_start = std::chrono::high_resolution_clock::now();
        alignParasail(algorithm, parasail_scores, query_ascii, db_ascii);
        auto parasail_end = std::chrono::high_resolution_clock::now();
        total_ms += parasail_end - parasail_start;
    }

    return total_ms / num_repeats;
}

// This runs our CPU implemenation n times and returns the average runtime
std::chrono::duration<double, std::milli> runCPU(int algorithm, const std::string &query_path, const std::vector<std::string> &sorted_db, const std::vector<int> &parasail_scores)
{
    // Load data
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> db_residues_cpu;
    std::vector<int> db_offsets_cpu;

    parseQuery(query_seq, query_path);
    generateDBCPU(sorted_db, db_residues_cpu, db_offsets_cpu);

    std::vector<int> cpu_scores(sorted_db.size());

    std::chrono::duration<double, std::milli> total_ms{};

    for (int i = 0; i < num_repeats; i++)
    {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        alignCPU(algorithm, cpu_scores, query_seq, db_residues_cpu, db_offsets_cpu);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        total_ms += cpu_end - cpu_start;

        checkResults(parasail_scores, cpu_scores, "CPU");
    }

    return total_ms / num_repeats;
}

// This runs our Naive GPU implemenation n times and returns the average runtime
std::chrono::duration<double, std::milli> runNaiveGPU(int algorithm, const std::string &query_path, const std::vector<std::string> &sorted_db, const std::vector<int> &parasail_scores)
{
    // Load data
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> db_residues_gpu;
    std::vector<int> db_offsets_gpu;

    parseQuery(query_seq, query_path);
    generateDBCPU(sorted_db, db_residues_gpu, db_offsets_gpu);

    std::vector<int> gpu_scores(sorted_db.size());

    float total_kernel_ms = 0.0f;

    for (int i = 0; i < num_repeats; i++)
    {
        total_kernel_ms += naiveInterGPU(algorithm, gpu_scores, query_seq, db_residues_gpu, db_offsets_gpu);
        checkResults(parasail_scores, gpu_scores, "Naive GPU");
    }

    return std::chrono::duration<double, std::milli>{total_kernel_ms / num_repeats};
}

// This runs our Optimised GPU hybrid implementation n times and returns the average runtime
std::chrono::duration<double, std::milli> runGPU(int algorithm, const std::string &query_path, const std::vector<std::string> &sorted_db, const std::vector<int> &parasail_scores, int threshold)
{
    // Data for GPU Implementation
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> small_seqs_soa;
    std::vector<int> soa_offsets;
    std::vector<std::vector<unsigned char>> large_seqs;

    // Load Queries
    parseQuery(query_seq, query_path);

    // Generate Memory Layouts
    generateDBGPU(sorted_db, small_seqs_soa, soa_offsets, large_seqs, threshold);

    std::vector<int> gpu_scores(sorted_db.size());

    size_t num_large_seqs = large_seqs.size();
    size_t num_small_seqs = sorted_db.size() - num_large_seqs;

    float total_kernel_ms = 0.0f;

    for (int i = 0; i < num_repeats; i++)
    {
        float inter_ms = 0.0f, intra_ms = 0.0f;
        if (num_small_seqs > 0)
            inter_ms = interAlignGPU(algorithm, gpu_scores, query_seq, small_seqs_soa, soa_offsets);
        if (num_large_seqs > 0)
            intra_ms = intraAlignGPU(algorithm, query_seq, large_seqs, gpu_scores, num_small_seqs);
        total_kernel_ms += inter_ms + intra_ms;

        checkResults(parasail_scores, gpu_scores, "GPU");
    }

    return std::chrono::duration<double, std::milli>{total_kernel_ms / num_repeats};
}

// This function displays the latency and GCUPS
void outputBenchmarks(std::chrono::duration<double, std::milli> total_ms, double total_cell_updates)
{
    double gcups = total_cell_updates / (total_ms.count() * 1e6);
    printf("%.1f %.2f\n", total_ms.count(), gcups);
}

// This program calculates the alignment scores for a query sequence against a database of target sequences using multiple different
// Implementations of NW and SW and measure their performance.
int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cerr << "Usage: ./bin/benchmark <algorithm> <num_seqs> <threshold> <query_path> <database_path>\n";
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
        std::cerr << "Usage: ./bin/benchmark <algorithm> <num_seqs> <threshold> <query_path> <database_path>\n";
        return 1;
    }
    catch (const std::out_of_range &e)
    {
        std::cerr << "Error: Number provided is out of integer range.\n";
        return 1;
    }

    const std::string query_path = argv[4]; // "./data/input/query_352.fasta";
    const std::string db_path = argv[5];    // "./data/input/uniprot_sprot.fasta";

    // Load or cache sorted db once
    std::vector<std::string>
        sorted_db = loadOrCacheDatabase(db_path, num_seqs);

    warmUpGpu();

    std::vector<int> parasail_scores(sorted_db.size());
    auto parasail_ms = runParasail(algorithm, query_path, sorted_db, parasail_scores);
    // auto cpu_ms = runCPU(algorithm, query_path, sorted_db, parasail_scores);
    //  auto naive_gpu_ms = runNaiveGPU(algorithm, query_path, sorted_db, parasail_scores);
    auto gpu_ms = runGPU(algorithm, query_path, sorted_db, parasail_scores, threshold);

    size_t total_db_residues = 0;
    for (const auto &seq : sorted_db)
        total_db_residues += seq.length();

    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, query_path);

    double total_cell_updates = static_cast<double>(query_seq.size()) * total_db_residues;

    outputBenchmarks(parasail_ms, total_cell_updates);
    // outputBenchmarks(naive_gpu_ms, total_cell_updates);
    // outputBenchmarks(cpu_ms, total_cell_updates);
    outputBenchmarks(gpu_ms, total_cell_updates);

    return 0;
}
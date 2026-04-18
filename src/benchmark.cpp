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

const std::string QUERY_PATH = "../data/input/query_360.fasta";
const std::string DB_PATH = "../data/input/uniprot_sprot.fasta";

void warmUpGpu()
{
    // This warms up the GPU for more accurate benchmarking
    std::vector<unsigned char> dummy_query = {0, 1, 2, 3}; // "ARND"
    std::vector<std::string> dummy_db = {"ARND"};
    std::vector<unsigned char> dummy_residues_soa{};
    std::vector<int> dummy_offsets{};
    std::vector<std::vector<unsigned char>> dummy_massive_sequences{};
    generateDBGPU(dummy_db, dummy_residues_soa, dummy_offsets, dummy_massive_sequences, 100000);
    std::vector<int> dummy_scores(1);

    interAlignGPU(0, dummy_scores, dummy_query, dummy_residues_soa, dummy_offsets);
}

std::chrono::duration<double, std::milli> runParasail(int algorithm, const std::vector<std::string> &sorted_db, std::vector<int> &parasail_scores, int num_repeats)
{
    // This will provide the gold standard used for verification
    std::string query_ascii;
    std::vector<std::string> db_ascii;
    parseQueryParasail(query_ascii, QUERY_PATH);
    generateDBParasail(sorted_db, db_ascii);

    std::chrono::duration<double, std::milli> total_ms;

    for (int i = 0; i < num_repeats; i++)
    {
        auto parasail_start = std::chrono::high_resolution_clock::now();
        alignParasail(algorithm, parasail_scores, query_ascii, db_ascii);
        auto parasail_end = std::chrono::high_resolution_clock::now();
        total_ms += parasail_end - parasail_start;
    }

    return total_ms / num_repeats;
}

std::chrono::duration<double, std::milli> runCPU(int algorithm, const std::vector<std::string> &sorted_db, const std::vector<int> &parasail_scores, int num_repeats)
{
    // Load data
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> db_residues_cpu;
    std::vector<int> db_offsets_cpu;

    parseQuery(query_seq, QUERY_PATH);
    generateDBCPU(sorted_db, db_residues_cpu, db_offsets_cpu);

    std::vector<int> cpu_scores(sorted_db.size());

    std::chrono::duration<double, std::milli> total_ms;

    for (int i = 0; i < num_repeats; i++)
    {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        alignCPU(algorithm, cpu_scores, query_seq, db_residues_cpu, db_offsets_cpu);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        total_ms += cpu_end - cpu_start;

        // VERIFY CORRECTNESS
        if (parasail_scores != cpu_scores)
            throw std::runtime_error("FAILED: CPU scores did not match Parasail!\n");
    }

    return total_ms / num_repeats;
}

std::chrono::duration<double, std::milli> runGPU(int algorithm, const std::vector<std::string> &sorted_db, const std::vector<int> &parasail_scores, int num_repeats, int threshold)
{
    // Data for GPU Implementation
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> small_seqs_soa;
    std::vector<int> soa_offsets;
    std::vector<std::vector<unsigned char>> large_seqs;

    // Load Queries
    parseQuery(query_seq, QUERY_PATH);

    // Generate Memory Layouts
    generateDBGPU(sorted_db, small_seqs_soa, soa_offsets, large_seqs, threshold);

    std::vector<int> gpu_scores(sorted_db.size());

    size_t num_large_seqs = large_seqs.size();
    size_t num_small_seqs = sorted_db.size() - num_large_seqs;

    std::chrono::duration<double, std::milli> total_ms;

    for (int i = 0; i < num_repeats; i++)
    {
        auto gpu_inter_start = std::chrono::high_resolution_clock::now();
        if (num_small_seqs > 0)
            interAlignGPU(algorithm, gpu_scores, query_seq, small_seqs_soa, soa_offsets);
        auto gpu_inter_end = std::chrono::high_resolution_clock::now();

        auto gpu_intra_start = std::chrono::high_resolution_clock::now();
        if (num_large_seqs > 0)
            intraAlignGPU(algorithm, query_seq, large_seqs, gpu_scores, num_small_seqs);
        auto gpu_intra_end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> gpu_inter_ms = gpu_inter_end - gpu_inter_start;
        std::chrono::duration<double, std::milli> gpu_intra_ms = gpu_intra_end - gpu_intra_start;
        total_ms += gpu_inter_ms + gpu_intra_ms;

        // VERIFY CORRECTNESS
        if (parasail_scores != gpu_scores)
            throw std::runtime_error("FAILED: GPU scores did not match Parasail!\n");
    }

    return total_ms / num_repeats;
}

void outputBenchmarks(std::chrono::duration<double, std::milli> total_ms, double total_cell_updates)
{
    double gcups = total_cell_updates / (total_ms.count() * 1e6);
    printf("%.1f %.1f\n", total_ms.count(), gcups);
}

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

    // Load or cache sorted db once
    std::vector<std::string> sorted_db = loadOrCacheDatabase(DB_PATH, num_seqs);

    warmUpGpu();

    std::vector<int> parasail_scores(sorted_db.size());
    auto parasail_ms = runParasail(algorithm, sorted_db, parasail_scores, 5);
    auto cpu_ms = runCPU(algorithm, sorted_db, parasail_scores, 5);
    auto gpu_ms = runGPU(algorithm, sorted_db, parasail_scores, 5, threshold);

    size_t total_db_residues = 0;
    for (const auto &seq : sorted_db)
        total_db_residues += seq.length();

    std::vector<unsigned char> query_seq;
    parseQuery(query_seq, QUERY_PATH);

    double total_cell_updates = static_cast<double>(query_seq.size()) * total_db_residues;

    outputBenchmarks(parasail_ms, total_cell_updates);
    outputBenchmarks(cpu_ms, total_cell_updates);
    outputBenchmarks(gpu_ms, total_cell_updates);

    return 0;
}
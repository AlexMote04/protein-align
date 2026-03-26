#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#include "parse.h"
#include "alignCPU.h"
#include "interAlignGPU.cuh"
#include "intraAlignGPU.cuh"
#include "alignParasail.h"

#define TILE_SIZE 32

const std::string QUERY_PATH = "../data/input/query";
const std::string DB_PATH = "../data/input/uniprot_sprot.fasta";

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

    int num_seqs = 0;
    int threshold = 0;

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
    // std::cout << "Warming up GPU context...\n";

    // Create a tiny dummy dataset
    std::vector<unsigned char> dummy_query = {0, 1, 2, 3}; // "ARND"
    std::vector<unsigned char> dummy_db = {0, 1, 2, 3};
    std::vector<int> dummy_offsets = {0, 4}; // 1 sequence, length 4
    std::vector<int> dummy_scores(1);

    interAlignGPU(algorithm, dummy_scores, dummy_query, dummy_db, dummy_offsets);

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

    std::cout << num_small_seqs << " " << num_big_seqs << "\n";
    auto gpu_inter_start = std::chrono::high_resolution_clock::now();
    if (num_small_seqs > 0)
    {
        interAlignGPU(algorithm, gpu_scores, query_seq, db_residues_soa, db_offsets);
    }
    auto gpu_inter_end = std::chrono::high_resolution_clock::now();

    // Final massive sequence with have largest length since db is sorted
    int max_target_len = 0;
    if (num_big_seqs > 0)
        max_target_len = static_cast<int>(massive_sequences[massive_sequences.size() - 1].size());

    int query_len = query_seq.size();
    int max_num_blocks_x = (query_len + TILE_SIZE - 1) / TILE_SIZE;
    int max_num_blocks_y = (max_target_len + TILE_SIZE - 1) / TILE_SIZE;
    int max_num_tiles_total = max_num_blocks_x * max_num_blocks_y;

    // Declare and Allocate device pointers
    unsigned char *d_query{}, *d_target{};
    int16_t *d_row_H{}, *d_row_E{}, *d_row_F{};
    int16_t *d_col_H{}, *d_col_E{}, *d_col_F{};
    int16_t *d_corner_H{};
    int *d_max_score{};

    if (num_big_seqs > 0)
    {
        cudaMalloc((void **)&d_query, query_len * sizeof(unsigned char));
        cudaMalloc((void **)&d_target, max_target_len * sizeof(unsigned char));

        cudaMalloc((void **)&d_row_H, max_target_len * sizeof(int16_t));
        cudaMalloc((void **)&d_row_E, max_target_len * sizeof(int16_t));
        cudaMalloc((void **)&d_row_F, max_target_len * sizeof(int16_t));

        cudaMalloc((void **)&d_col_H, query_len * sizeof(int16_t));
        cudaMalloc((void **)&d_col_E, query_len * sizeof(int16_t));
        cudaMalloc((void **)&d_col_F, query_len * sizeof(int16_t));

        cudaMalloc((void **)&d_corner_H, max_num_tiles_total * sizeof(int16_t));
        cudaMalloc((void **)&d_max_score, sizeof(int));

        // Copy the Query sequence ONCE
        cudaMemcpy(d_query, query_seq.data(), query_len * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    auto gpu_intra_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < massive_sequences.size(); i++)
    {
        gpu_scores[i + num_small_seqs] = intraAlignGPU(
            algorithm, query_seq, massive_sequences[i],
            d_query, d_target,
            d_row_H, d_row_E, d_row_F,
            d_col_H, d_col_E, d_col_F,
            d_corner_H, d_max_score);
    }

    if (num_big_seqs > 0)
    {
        cudaFree(d_query);
        cudaFree(d_target);
        cudaFree(d_row_H);
        cudaFree(d_row_E);
        cudaFree(d_row_F);
        cudaFree(d_col_H);
        cudaFree(d_col_E);
        cudaFree(d_col_F);
        cudaFree(d_corner_H);
        cudaFree(d_max_score);
    }

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
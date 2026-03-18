#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#include "parse.h"
#include "alignCPU.h"
#include "alignGPU.cuh"
#include "alignParasail.h"

const std::string QUERY_PATH = "../data/input/query";
const std::string DB_PATH = "../data/input/fixed_len.fasta";

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./bin/benchmark <algorithm> <num_seqs>\n";
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

    int num_seqs = std::atoi(argv[2]);

    initConversionTable();

    // Data for Custom Implementation
    std::vector<unsigned char> query_seq;
    std::vector<unsigned char> db_residues;
    std::vector<unsigned char> db_residues_soa;
    std::vector<int> db_offsets;

    // Data for Parasail (ASCII)
    std::string query_ascii;
    std::vector<std::string> db_ascii;

    try
    {
        // Load Converted Data
        parseQuery(query_seq, QUERY_PATH);
        parseDB(db_residues, db_offsets, DB_PATH, num_seqs);

        // SoA residues for optimised kernel
        parseDBSoA(db_residues_soa, db_offsets, DB_PATH, num_seqs);

        // Load ASCII Data
        parseQueryParasail(query_ascii, QUERY_PATH);
        parseDBParasail(db_ascii, DB_PATH, num_seqs);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Parsing Error: " << e.what() << '\n';
        return 1;
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

    alignGPU(algorithm, dummy_scores, dummy_query, dummy_db, dummy_offsets);

    std::vector<int> cpu_scores(num_seqs);
    std::vector<int> parasail_scores(num_seqs);
    std::vector<int> gpu_scores(num_seqs);

    // ==========================================
    // TIMING PHASE
    // ==========================================
    // std::cout << "Running Naive CPU...\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();
    alignCPU(algorithm, cpu_scores, query_seq, db_residues, db_offsets);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // std::cout << "Running Parasail (SIMD)...\n";
    auto parasail_start = std::chrono::high_resolution_clock::now();
    alignParasail(algorithm, parasail_scores, query_ascii, db_ascii);
    auto parasail_end = std::chrono::high_resolution_clock::now();

    // std::cout << "Running Optimized GPU...\n";
    auto gpu_start = std::chrono::high_resolution_clock::now();
    alignGPU(algorithm, gpu_scores, query_seq, db_residues_soa, db_offsets);
    auto gpu_end = std::chrono::high_resolution_clock::now();

    // ==========================================
    // VERIFICATION PHASE
    // ==========================================
    int cpu_mismatches = 0;
    for (size_t i = 0; i < num_seqs; i++)
    {
        if (cpu_scores[i] != parasail_scores[i])
        {
            if (cpu_mismatches < 5)
            { // Only print the first 5 to avoid terminal spam
                std::cerr << "--- MISMATCH AT SEQUENCE " << i << " ---\n";
                std::cerr << "Target String  : " << db_ascii[i] << "\n";
                std::cerr << "Target Length  : " << db_ascii[i].length() << " residues\n";
                std::cerr << "Naive CPU Score: " << cpu_scores[i] << "\n";
                std::cerr << "Parasail Score : " << parasail_scores[i] << "\n";
                std::cerr << "Difference     : " << std::abs(cpu_scores[i] - parasail_scores[i]) << "\n\n";
            }
            cpu_mismatches++;
        }
    }

    if (cpu_mismatches > 0)
    {
        std::cerr << "FAILED: " << cpu_mismatches << " Naive CPU scores did not match Parasail!\n";
        return 1;
    }

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
                std::cerr << "Naive CPU Score: " << gpu_scores[i] << "\n";
                std::cerr << "Parasail Score : " << parasail_scores[i] << "\n";
                std::cerr << "Difference     : " << std::abs(gpu_scores[i] - parasail_scores[i]) << "\n\n";
            }
            gpu_mismatches++;
        }
    }

    if (gpu_mismatches > 0)
    {
        std::cerr << "FAILED: " << gpu_mismatches << " Naive CPU scores did not match Parasail!\n";
        return 1;
    }

    // std::cout << "PASS! All implementations perfectly match the Gold Standard.\n\n";

    // ==========================================
    // METRICS & REPORTING (Console)
    // ==========================================
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;
    std::chrono::duration<double, std::milli> parasail_ms = parasail_end - parasail_start;
    std::chrono::duration<double, std::milli> gpu_ms = gpu_end - gpu_start;

    double total_cell_updates = static_cast<double>(query_seq.size()) * db_residues.size();
    double cpu_gcups = total_cell_updates / (cpu_ms.count() * 1e6);
    double parasail_gcups = total_cell_updates / (parasail_ms.count() * 1e6);
    double gpu_gcups = total_cell_updates / (gpu_ms.count() * 1e6);

    // Latency: CPU, PARASAIL, OPTMISED GPU. THROUGHPUT CPU, PARASAIL, OPTIMISED GPU. SPEEDUP: CPU vs. Optimised, Parasail vs. Optimised
    printf("%.1f, %.1f, %.1f, ", cpu_ms.count(), parasail_ms.count(), gpu_ms.count());
    printf("%.2f, %.2f, %.2f, ", cpu_gcups, parasail_gcups, gpu_gcups);
    printf("%.1f, %.1f\n", cpu_ms.count() / gpu_ms.count(), parasail_ms.count() / gpu_ms.count());

    // ==========================================
    // METRICS & REPORTING (Verbose)
    // ==========================================
    // printf("--- Execution Time ---\n");
    // printf("Naive CPU: %.2f ms\n", cpu_ms.count());
    // printf("Parasail : %.2f ms\n", parasail_ms.count());
    // printf("GPU CUDA : %.2f ms\n\n", gpu_ms.count());

    // printf("--- GCUPS (Giga Cell Updates Per Second) ---\n");
    // printf("Naive CPU: %.4f\n", cpu_gcups);
    // printf("Parasail : %.4f\n", parasail_gcups);
    // printf("GPU CUDA : %.4f\n\n", gpu_gcups);

    // printf("--- Speedup ---\n");
    // printf("GPU vs Naive CPU: %.2fx\n", cpu_ms.count() / gpu_ms.count());
    // printf("GPU vs Parasail : %.2fx\n", parasail_ms.count() / gpu_ms.count());

    return 0;
}
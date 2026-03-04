#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#include "parse.h"
#include "sw_cpu.h"
#include "sw_gpu.cuh"

const std::string QUERY_PATH = "../data/input/query";
const std::string DB_PATH = "../data/input/uniprot_sprot_sorted";

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: ./bin/benchmark <num_seqs>" << std::endl;
    return 1;
  }

  int num_seqs = std::atoi(argv[1]);

  if (num_seqs < 1 || num_seqs > MAX_NUM_SEQS)
  {
    std::cerr << "Error: num_seqs must be between 1 and " << MAX_NUM_SEQS << std::endl;
    return 1;
  }

  initConversionTable();                // Maps amino acids to numbers
  std::vector<unsigned char> query_seq; // Query sequence
  std::vector<unsigned char> db_residues;
  std::vector<int> db_offsets;

  try
  {
    parseQuery(query_seq, QUERY_PATH);
    parseDB(db_residues, db_offsets, DB_PATH, num_seqs);
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }

  std::vector<int> cpu_scores(db_offsets.size() - 1); // Pre-allocate scores vector
  std::vector<int> gpu_scores(db_offsets.size() - 1);

  auto cpu_start = std::chrono::high_resolution_clock::now();
  swCPU(cpu_scores, query_seq, db_residues, db_offsets);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  auto gpu_start = std::chrono::high_resolution_clock::now();
  swGPU(gpu_scores, query_seq, db_residues, db_offsets);
  auto gpu_end = std::chrono::high_resolution_clock::now();

  if (cpu_scores != gpu_scores)
  {
    std::cout << "MISMATCH!" << std::endl;
    return 1;
  }
  std::cout << "PASS!" << std::endl;

  std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;
  std::chrono::duration<double, std::milli> gpu_ms = gpu_end - gpu_start;

  double total_cell_updates = static_cast<double>(query_seq.size()) * db_residues.size();

  printf("CPU Total: %.2f ms\n", cpu_ms.count());
  printf("GPU Total (incl. Transfer): %.2f ms\n", gpu_ms.count());

  double cpu_gcups = total_cell_updates / (cpu_ms.count() * 1e6);
  double gpu_gcups = total_cell_updates / (gpu_ms.count() * 1e6);

  printf("CPU GCUPS: %.4f\n", cpu_gcups);
  printf("GPU GCUPS: %.4f\n", gpu_gcups);
  printf("Speedup: %.2fx\n", cpu_ms.count() / gpu_ms.count());

  return 0;
}
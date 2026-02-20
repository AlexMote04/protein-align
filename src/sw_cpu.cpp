#include "sw_cpu.h"
#include <iostream>

int align(const unsigned char* query, const unsigned char* db, const int M, const int N){
  int16_t max_score = 0;          // Keep track of optimal alignment score

  // Initialise matrices
  std::vector<int16_t> H(M * N);  // 16 bit signed integers
  std::vector<int16_t> E(M * N);
  std::vector<int16_t> F(M * N);

  // Set first row
  for(int i = 0; i < N; i += 1){
    H[i] = 0;
    F[i] = -10000; // Large negative number
  }
  // Set first column
  for(int j = 0; j < M * N; j += N){
    H[j] = 0;
    E[j] = -10000;
  }

  // Fill matrices
  for(int i = 1; i < M; i++){ // Skip first row
    for(int j = 1; j < N; j++){ // Skip first column
      int index_1d = i * N + j;
      E[index_1d] = std::max(H[index_1d - 1] - OPEN, E[index_1d - 1] - EXTEND); // Horizontal gap
      F[index_1d] = std::max(H[index_1d - N] - OPEN, F[index_1d - N] - EXTEND); // Vertical gap

      int16_t val = H[index_1d - N - 1] + blosum62[query[i-1] * 24 + db[j-1]]; // Diag Score

      // Find max of scores
      val = E[index_1d] > val ? E[index_1d] : val;
      val = F[index_1d] > val ? F[index_1d] : val;
      H[index_1d] = (val < 0) ? 0 : val;

      max_score = std::max(max_score, H[index_1d]); // Update best score seen so far
    }
  }
  return max_score;
}

int swCPU(std::vector<int> &scores, const std::vector<unsigned char> &query, const std::vector<unsigned char> &db, const std::vector<int> &offsets){
  const int M = query.size() + 1; // Number of rows in matrix
  int N; // Number of columns in matrix
  int db_size = offsets.size(); // Number of sequences in database

  for(int i = 0; i < db_size; i++){ // Loop through each sequence in db
    if(i == (db_size - 1)){
      N = (db.size() - offsets[i]) + 1; // Last sequence spans from final offset to length of db
    } else {
      N = (offsets[i+1] - offsets[i]) + 1; // Length of sequence is difference between offsets
    }
    scores[i] = align(query.data(), db.data() + offsets[i], M, N);
  }
  return 0;
}
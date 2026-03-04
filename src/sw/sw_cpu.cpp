#include "sw_cpu.h"

int align_sw(const unsigned char *query_seq, const unsigned char *target_seq, const int M, const int N, std::vector<int16_t> &H, std::vector<int16_t> &E, std::vector<int16_t> &F)
{
  int16_t max_score = 0; // Keep track of optimal alignment score

  // Set first row
  for (int i = 0; i < N; i += 1)
  {
    H[i] = 0;
    F[i] = -10000;
  }
  // Set first column
  for (int j = 0; j < M * N; j += N)
  {
    H[j] = 0;
    E[j] = -10000;
  }

  // Fill matrices
  for (int i = 1; i < M; i++)
  { // Skip first row
    for (int j = 1; j < N; j++)
    { // Skip first column
      int idx = i * N + j;
      E[idx] = std::max(H[idx - 1] - OPEN, E[idx - 1] - EXTEND);                          // Horizontal gap
      F[idx] = std::max(H[idx - N] - OPEN, F[idx - N] - EXTEND);                          // Vertical gap
      int16_t val = H[idx - N - 1] + blosum62[query_seq[i - 1] * 24 + target_seq[j - 1]]; // Substitution

      // Find max of scores
      val = std::max(E[idx], val);
      val = std::max(F[idx], val);
      H[idx] = std::max(val, (int16_t)0);

      max_score = std::max(max_score, H[idx]); // Update best score seen so far
    }
  }
  return max_score;
}

int swCPU(std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets)
{
  // Find the "High-Water Mark" (the longest sequence in the database)
  int max_residue = 0;
  for (size_t i = 0; i < scores.size(); i++)
  {
    int current_residue = db_offsets[i + 1] - db_offsets[i];
    if (current_residue > max_residue)
    {
      max_residue = current_residue;
    }
  }

  const int M = query_seq.size() + 1;
  const int N_max = max_residue + 1;

  // Allocate memory buffers EXACTLY ONCE
  std::vector<int16_t> H(M * N_max);
  std::vector<int16_t> E(M * N_max);
  std::vector<int16_t> F(M * N_max);

  int N;
  for (int i = 0; i < scores.size(); i++)
  {                                            // Loop through each sequence in db
    N = db_offsets[i + 1] - db_offsets[i] + 1; // Length of sequence is the difference between offsets
    scores[i] = align_sw(query_seq.data(), db_residues.data() + db_offsets[i], M, N, H, E, F);
  }
  return 0;
}
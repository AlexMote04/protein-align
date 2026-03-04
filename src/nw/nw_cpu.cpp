#include "nw_cpu.h"

int align_nw(const unsigned char *query_seq, const unsigned char *target_seq, const int M, const int N, std::vector<int16_t> &H, std::vector<int16_t> &E, std::vector<int16_t> &F)
{
  // Initialize top-left corner
  H[0] = 0;
  E[0] = -10000;
  F[0] = -10000;

  // Initialize first row (Horizontal gaps)
  for (int j = 1; j < N; j++)
  {
    H[j] = -OPEN - (j - 1) * EXTEND;
    F[j] = -10000; // Vertical gap impossible here
  }

  // Initialize first column (Vertical gaps)
  for (int i = 1; i < M; i++)
  {
    int idx = i * N;
    H[idx] = -OPEN - (i - 1) * EXTEND;
    E[idx] = -10000; // Horizontal gap impossible here
  }

  // Fill matrices
  for (int i = 1; i < M; i++)
  {
    for (int j = 1; j < N; j++)
    {
      int idx = i * N + j;
      E[idx] = std::max((int)H[idx - 1] - OPEN, (int)E[idx - 1] - EXTEND);
      F[idx] = std::max((int)H[idx - N] - OPEN, (int)F[idx - N] - EXTEND);
      int16_t val = H[idx - N - 1] + blosum62[query_seq[i - 1] * 24 + target_seq[j - 1]];

      val = std::max((int)E[idx], (int)val);
      val = std::max((int)F[idx], (int)val);
      H[idx] = val;
    }
  }
  return H[M * N - 1];
}

int nwCPU(std::vector<int> &scores, const std::vector<unsigned char> &query_seq, const std::vector<unsigned char> &db_residues, const std::vector<int> &db_offsets)
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
    scores[i] = align_nw(query_seq.data(), db_residues.data() + db_offsets[i], M, N, H, E, F);
  }
  return 0;
}
# protein-align

A CUDA-accelerated protein sequence database search tool implementing pairwise sequence alignment at scale. Given a single query sequence and a database of target sequences, the tool dispatches thousands of independent pairwise alignments to the GPU in parallel. Both the classic **Needleman-Wunsch** (global) and **Smith-Waterman** (local) algorithms are supported, with CPU and GPU backends available for comparison.

Developed as a third-year Computer Science project.

---

## Algorithms

### Needleman-Wunsch (Global Alignment)
Computes the optimal global alignment between two sequences using dynamic programming. Suitable for sequences of similar length and biological origin.

### Smith-Waterman (Local Alignment)
Computes the optimal local alignment, identifying the highest-scoring subsequence pair. The industry standard for sensitive homology search.

---

## Parallelism Strategies

The GPU backend splits the workload based on sequence length, implementing two distinct levels of parallelism:

### Inter-Alignment Parallelism (Small Sequences)

For sequences below a specified length threshold, the tool processes multiple alignments simultaneously. Sequences are grouped into batches and stored in a padded Structure of Arrays (SoA) layout in memory. Each CUDA thread is assigned a single pairwise alignment and computes the full DP matrix independently. This achieves massive throughput and high occupancy when the database contains many small targets.

### Intra-Alignment Parallelism (Massive Sequences)

For extremely long sequences that exceed the threshold, the DP matrix is too large for a single thread. The tool uses a tiled wavefront decomposition approach. It exploits the anti-diagonal dependency structure of the DP matrix, where cells on the same anti-diagonal are independent and can be computed simultaneously.
* **Macro-level wavefront:** The overall DP matrix is processed in blocks, mapping directly to CUDA thread blocks.
* **Micro-level wavefront:** Inside each block, threads step through anti-diagonals within a local tile, pipelining computation and syncing at the warp level.

---

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| CUDA Toolkit | ≥ 11.0 | GPU compilation and runtime |
| CMake | ≥ 3.18 | Build system |
| C++17 / CUDA 17 | — | Host and device code standards |
| [Parasail](https://github.com/jeffdaily/parasail) | v2.6.2 | CPU baseline (fetched automatically via CMake) |

Parasail is fetched and built automatically at configure time — no manual installation required.

---

## Build

```bash
mkdir bin
cd bin
cmake ..
cmake --build .
```

CMake will detect your local GPU architecture automatically via CMAKE_CUDA_ARCHITECTURES native.

---

## Usage
```bash
./benchmark <nw|sw> <number_of_alignments> <threshold>
```

| Argument | Description |
|---|---|
| nw/sw | Alignment algorithm — Needleman-Wunsch or Smith-Waterman | 
| <number_of_alignments> | Number of pairwise alignments to perform against the query |
| <threshold> | Sequence length cutoff. Sequences longer than this trigger the wavefront algorithm |

### Examples
```bash
# Run 10,000 Smith-Waterman alignments, using wavefront for sequences > 500 residues
./benchmark sw 10000 500

# Run 5,000 Needleman-Wunsch alignments, using wavefront for sequences > 1000 residues
./benchmark nw 5000 1000
```

## Benchmarks
TODO:
Results collected on [GPU model], CUDA [version], with sequences of average length [L].

Implementation | Time (ms) | GCUPS | Speedup
|---|---|---|---|
|CPU (Parasail SIMD) | | | 1.0x |
|GPU (Hybrid Inter/Intra) | | | |

GCUPS = Giga Cell Updates Per Second — the standard throughput metric for sequence alignment, computed as (query_len × target_len × num_alignments) / (time_s × 10⁹).

## Project Structure
```text
protein-align/
├── include/              # Header files
├── src/
│   ├── benchmark.cpp     # Benchmark entry point and validation
│   ├── parse.cpp         # FASTA sequence parsing and memory layout generation
│   ├── alignCPU.cpp      # Naive CPU reference implementation
│   ├── alignParasail.cpp # Parasail-backed SIMD CPU baseline
│   ├── interAlignGPU.cu  # CUDA kernel for independent small sequences
│   └── intraAlignGPU.cu  # CUDA kernel for tiled wavefront massive sequences
├── test/                 # Catch2 unit tests (build target: run_tests)
├── CMakeLists.txt
└── README.md
```

## License
MIT
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

The GPU backend implements two levels of parallelism:

### Intra-thread Parallelism — One Thread per Alignment
Each CUDA thread is assigned a single pairwise alignment and computes the full DP matrix independently. This approach maps naturally to the database search workload, where alignments are entirely independent, and achieves high occupancy when the number of alignments is large relative to sequence length.

### Inter-thread Parallelism — Wavefront Decomposition
Exploits the anti-diagonal dependency structure of the DP recurrence. Cells on the same anti-diagonal are independent and can be computed simultaneously.

- **Block-level wavefront:** Each anti-diagonal of the DP matrix is processed by a full CUDA block, with threads within the block handling cells along that diagonal in parallel.
- **Micro-level wavefront (within-block):** A finer-grained wavefront is applied inside each block, pipelining computation across warps to maximise arithmetic throughput and hide memory latency.

---

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| CUDA Toolkit | ≥ 11.0 | GPU compilation and runtime |
| CMake | ≥ 3.14 | Build system |
| C++14 | — | Host code standard |
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

CMake will detect your local GPU architecture automatically via `CMAKE_CUDA_ARCHITECTURES native`.

---

## Usage

```bash
./benchmark <nw|sw> <number_of_alignments>
```

| Argument | Description |
|---|---|
| `nw` \| `sw` | Alignment algorithm — Needleman-Wunsch or Smith-Waterman |
| `<number_of_alignments>` | Number of pairwise alignments to perform against the query |

### Examples

```bash
# Run 10,000 Smith-Waterman alignments
./benchmark sw 10000

# Run 5,000 Needleman-Wunsch alignments
./benchmark nw 5000
```

---

## Benchmarks

> Results collected on **[GPU model]**, CUDA **[version]**, with sequences of average length **[L]**.

### Smith-Waterman

| Implementation | Alignments | Time (ms) | GCUPS |
|---|---|---|---|
| CPU (scalar) | | | |
| CPU (Parasail SIMD) | | | |
| GPU — intra-thread | | | |
| GPU — inter-thread (wavefront blocks) | | | |
| GPU — inter-thread (micro wavefront) | | | |

### Needleman-Wunsch

| Implementation | Alignments | Time (ms) | GCUPS |
|---|---|---|---|
| CPU (scalar) | | | |
| CPU (Parasail SIMD) | | | |
| GPU — intra-thread | | | |
| GPU — inter-thread (wavefront blocks) | | | |
| GPU — inter-thread (micro wavefront) | | | |

> **GCUPS** = Giga Cell Updates Per Second — the standard throughput metric for sequence alignment, computed as `(query_len × target_len × num_alignments) / (time_s × 10⁹)`.

---

## Project Structure

```
protein-align/
├── include/              # Header files
├── src/
│   ├── benchmark.cpp         # Benchmark entry point
│   ├── parse.cpp             # Sequence parsing
│   ├── alignCPU.cpp          # CPU reference implementation
│   ├── alignParasail.cpp     # Parasail-backed CPU baseline
│   └── alignGPUOptimised.cu  # CUDA kernels (intra- and inter-thread)
├── test/                 # Unit tests (build target: run_tests)
├── CMakeLists.txt
└── README.md
```

---

## License

MIT
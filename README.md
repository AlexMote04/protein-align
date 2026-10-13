# protein-align

CUDA protein database search. Given one query sequence and a protein database (e.g. UniProt Swiss-Prot), it computes the optimal alignment score of the query against every database sequence on the GPU. It supports **Needleman-Wunsch** (global) and **Smith-Waterman** (local) alignment with affine gap penalties and BLOSUM62 scoring (gap open 10, extend 1). Every GPU score is checked against [Parasail](https://github.com/jeffdaily/parasail), a SIMD CPU alignment library, on every run.

Third-year Computer Science project.

## Results

Full Swiss-Prot database (~574k sequences) with synthetic queries of 64 to 4096 residues. GPU times are kernel time; the CPU baseline is single-threaded Parasail (`scan` profile, saturating 8/16-bit arithmetic). Raw data is in [results/](results/).

| Query length | NW: Parasail | NW: GPU hybrid | NW speedup | SW: Parasail | SW: GPU hybrid | SW speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 0.83 GCUPS | 10.75 GCUPS | 13.0× | 1.32 GCUPS  | 10.99 GCUPS | 8.3× |
| 256  | 2.19 GCUPS | 12.79 GCUPS | 5.8×  | 4.18 GCUPS  | 12.85 GCUPS | 3.1× |
| 1024 | 3.64 GCUPS | 13.34 GCUPS | 3.7×  | 8.19 GCUPS  | 13.37 GCUPS | 1.6× |
| 4096 | 4.22 GCUPS | 13.41 GCUPS | 3.2×  | 10.52 GCUPS | 13.41 GCUPS | 1.3× |

GCUPS (giga cell updates per second) = `query_len × total_db_residues / (time_s × 10⁹)`.

Other findings from the experiments:

- **Optimisation ablation:** the optimised inter-sequence kernel runs at 6.35 GCUPS (NW) and 8.99 GCUPS (SW). The naive one-thread-per-alignment kernel runs at 0.65 GCUPS, so the optimised kernel is 10–14× faster.
- **Inter/intra crossover:** with one thread per alignment, very long targets leave most of the GPU idle. Once targets reach about 2048 residues, the tiled wavefront kernel overtakes the inter-sequence kernel.
- **Hybrid threshold:** routing sequences longer than 8192 residues to the wavefront kernel gives 12–13 GCUPS. Pure inter-sequence routing gives about 9 GCUPS.
- **Database composition:** on databases of very short uniform-length sequences, the inter kernel reaches up to 78 GCUPS.

## How it works

### Data preparation (`parse.cpp`)

- The database FASTA is parsed once, sorted by length and cached as a binary file (`<db>.fasta.bin`). Later runs skip the parse.
- Residues are encoded to BLOSUM62 indices.
- Sequences shorter than the threshold are packed in batches of 32 into a padded, interleaved **structure-of-arrays** layout. Each warp reads its 32 sequences with coalesced loads. Because the database is sorted, sequences in a batch have similar lengths, which keeps padding small.
- Sequences at or above the threshold are routed to the intra-sequence kernel.

### Inter-sequence kernel (`interAlignGPU.cu`)

- One thread computes one whole alignment. It keeps two rolling DP rows in global memory as packed `short2` (H, F) values, interleaved across the 32 lanes of a batch so that warp accesses coalesce.
- BLOSUM62 is staged in shared memory, and the query is read from constant memory.
- The first pass uses **int16** arithmetic. A thread flags its alignment if any score nears the int16 limit. The host then re-runs only the batches that contain a flagged alignment, using an int32 kernel. The fast int16 path covers the common case, and the scores stay exact.

### Intra-sequence kernel (`intraAlignGPU.cu`)

- Used for very long targets, where a single thread per alignment would be far too slow.
- The DP matrix is split into 32×32 tiles and processed as a **two-level wavefront**:
  - the host launches one kernel per anti-diagonal of tiles;
  - inside each tile, a 32-thread block sweeps the cell anti-diagonals in shared memory.
- Tile boundary rows and columns are passed between tiles through global memory.
- Alignments are spread across **4 CUDA streams**, with pinned host buffers and async copies, so transfers overlap compute.
- The same int16-first, int32-fallback scheme is used here.

### Baselines and verification

- `alignCPU.cpp`: a straightforward sequential CPU implementation.
- `naiveInterGPU.cu`: a naive GPU kernel used for the ablation.
- `alignParasail.cpp`: the Parasail baseline.
- `benchmark.cpp` runs each implementation 3 times after a GPU warm-up. It throws an error if any score differs from Parasail's.

## Build

Requirements:

- CUDA Toolkit ≥ 11.0
- CMake ≥ 3.24
- a C++17 compiler
- OpenMP

Parasail v2.6.2 is fetched and built automatically by CMake. The GPU architecture is detected with `CMAKE_CUDA_ARCHITECTURES native`.

```bash
./build.sh                         # or: cmake -S . -B bin && cmake --build bin -j
```

This produces `bin/benchmark` and `bin/run_tests` (Catch2 unit tests for the FASTA parser and the SoA layout generator).

## Usage

```bash
./bin/benchmark <nw|sw> <num_seqs> <threshold> <query.fasta> <database.fasta>

# e.g. full Swiss-Prot, SW, wavefront kernel for targets >= 8192 residues
./bin/benchmark sw 600000 8192 data/input/query_464.fasta data/input/uniprot_sprot.fasta
```

| Argument | Meaning |
|---|---|
| `nw` / `sw` | Needleman-Wunsch or Smith-Waterman |
| `num_seqs` | Maximum number of database sequences to load |
| `threshold` | Sequences with length ≥ threshold go to the intra-sequence (wavefront) kernel |

The program prints `latency_ms GCUPS` on two lines: Parasail first, then the GPU hybrid.

## Reproducing the experiments

```bash
mkdir -p data/input
# Swiss-Prot: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip -c uniprot_sprot.fasta.gz > data/input/uniprot_sprot.fasta
python3 scripts/generate_database.py                       # random 464-residue query
python3 scripts/generate_subset.py ...                     # random Swiss-Prot subsets (see --help)
pip install tqdm
python3 scripts/experiment_3.py                            # writes results/experiment_3.csv
```

| Script | Experiment |
|---|---|
| `experiment_1.py` | Sequential CPU vs Parasail vs GPU inter-sequence |
| `experiment_2.py` | Naive vs optimised inter-sequence kernel |
| `experiment_3.py` | Query-length sensitivity (inter-sequence only) |
| `experiment_4.py` | Database composition (uniform-length synthetic databases vs Swiss-Prot) |
| `experiment_5.py` | Long-sequence regime: inter vs intra crossover |
| `experiment_6.py` | Hybrid threshold sweep |
| `experiment_7.py` | Final hybrid vs Parasail across query lengths |

Experiments 1 and 2 also need the CPU and naive GPU runs. These are currently commented out in `main()` in `src/benchmark.cpp`.

## Project structure

```text
protein-align/
├── include/               # Headers (params.h: scoring constants; blosum62.h; catch.hpp)
├── src/
│   ├── benchmark.cpp      # Entry point: loads data, runs + verifies each implementation
│   ├── parse.cpp          # FASTA parsing, binary DB cache, SoA batch layout
│   ├── interAlignGPU.cu   # One-thread-per-alignment kernels (int16 + int32 fallback)
│   ├── intraAlignGPU.cu   # Tiled wavefront kernels for long sequences, multi-stream
│   ├── naiveInterGPU.cu   # Naive GPU kernel (ablation baseline)
│   ├── alignCPU.cpp       # Sequential CPU reference
│   └── alignParasail.cpp  # Parasail SIMD baseline
├── test/                  # Catch2 tests (target: run_tests)
├── scripts/               # Experiment drivers + data generation
├── results/               # Experiment CSVs
├── build.sh
└── CMakeLists.txt
```

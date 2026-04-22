import csv
import subprocess
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Optimisation Ablation — Long-Sequence Regime (Inter vs Intra Crossover)

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK = PROJECT_ROOT / "bin" / "benchmark"

NUM_SEQS = 100
ALGORITHMS = ["nw", "sw"]
QUERY_LENGTH = 464

INPUT_DIR = PROJECT_ROOT / "data" / "input"
QUERY_FILE = INPUT_DIR / f"query_{QUERY_LENGTH}.fasta"
OUTPUT_CSV = PROJECT_ROOT / "results" / "experiment_5.csv"

# The lengths we want to test to find the crossover point
SEQUENCE_LENGTHS = [64 * 2**i for i in range(10)]

# Define thresholds to force routing
THRESHOLDS = {
    "GPU Inter": 1_000_000,  # High threshold routes everything to Inter
    "GPU Intra": 1,  # 1 threshold routes everything to Intra
}

# The C++ benchmark outputs two lines: Parasail, then GPU
KEYS = ["parasail", "gpu"]


def generate_random_fasta(path: Path, length: int, count: int) -> None:
    """Generates a FASTA file with random sequences of a specific length."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    with path.open("w") as f:
        for i in range(count):
            seq = "".join(random.choices(amino_acids, k=length))
            f.write(f">seq_{i}_len_{length}\n{seq}\n")


def prepare_databases() -> dict[int, Path]:
    """Generates synthetic datasets for each sequence length if they don't exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_paths = {}

    for length in SEQUENCE_LENGTHS:
        db_path = INPUT_DIR / f"exp5_len_{length}.fasta"
        db_paths[length] = db_path
        if not db_path.exists():
            print(f"Generating 100-sequence database at length {length}...")
            generate_random_fasta(db_path, length, NUM_SEQS)

    return db_paths


def run_benchmark(
    algo: str, database_path: Path, threshold: int
) -> dict[str, dict[str, float]]:
    """Runs the benchmark and returns latency and GCUPS."""
    cmd = [
        str(BENCHMARK),
        algo,
        str(NUM_SEQS),
        str(threshold),
        str(QUERY_FILE),
        str(database_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Benchmark failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = proc.stdout.strip().splitlines()

    if len(lines) != len(KEYS):
        print(
            f"\n[ERROR] Expected {len(KEYS)} output lines, got {len(lines)}: {proc.stdout!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = {}
    for key, line in zip(KEYS, lines):
        try:
            ms, gcups = map(float, line.split())
        except ValueError:
            print(f"\n[ERROR] Could not parse line {line!r}", file=sys.stderr)
            sys.exit(1)
        result[key] = {"latency_ms": ms, "gcups": gcups}
    return result


def write_csv(results: list[dict]) -> None:
    """Writes the flat list of result dictionaries to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Algorithm", "Sequence Length", "Implementation", "Latency (ms)", "GCUPS"]
        )
        for row in results:
            writer.writerow(
                [
                    row["algorithm"].upper(),
                    row["sequence_length"],
                    row["implementation"],
                    f"{row['latency_ms']:.2f}",
                    f"{row['gcups']:.3f}",
                ]
            )
    print(f"Results written to {OUTPUT_CSV}")


def main() -> None:
    if not BENCHMARK.exists():
        print(f"[ERROR] Binary not found: {BENCHMARK}", file=sys.stderr)
        print("Build the project first (cmake + make).", file=sys.stderr)
        sys.exit(1)

    if not QUERY_FILE.exists():
        print(f"[ERROR] Query file not found: {QUERY_FILE}", file=sys.stderr)
        print(
            "Ensure you have generated the query file from the previous experiment.",
            file=sys.stderr,
        )
        sys.exit(1)

    db_paths = prepare_databases()

    # Flat list to store results for easy CSV writing
    csv_data = []

    # Tasks: Algorithm x Sequence Length x Strategy (Inter vs Intra)
    tasks = [
        (algo, length, strategy, threshold)
        for algo in ALGORITHMS
        for length in SEQUENCE_LENGTHS
        for strategy, threshold in THRESHOLDS.items()
    ]

    print(f"\nStarting {len(tasks)} benchmark tasks to find the crossover point.")

    progress_bar = tqdm(tasks, total=len(tasks), unit="run")

    for algo, length, strategy, threshold in progress_bar:
        progress_bar.set_description(f"{algo.upper()} | Len: {length} | {strategy}")

        run_results = run_benchmark(algo, db_paths[length], threshold)

        # Append GPU result (labeled as Inter or Intra based on the loop)
        csv_data.append(
            {
                "algorithm": algo,
                "sequence_length": length,
                "implementation": strategy,
                "latency_ms": run_results["gpu"]["latency_ms"],
                "gcups": run_results["gpu"]["gcups"],
            }
        )

        # We also get Parasail for free on every run.
        # To avoid duplicate Parasail lines in the plot, we only append it during the Inter run.
        if strategy == "GPU Inter":
            csv_data.append(
                {
                    "algorithm": algo,
                    "sequence_length": length,
                    "implementation": "Parasail",
                    "latency_ms": run_results["parasail"]["latency_ms"],
                    "gcups": run_results["parasail"]["gcups"],
                }
            )

    print("\nAll benchmarks complete!")
    write_csv(csv_data)


if __name__ == "__main__":
    main()

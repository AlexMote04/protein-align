import csv
import subprocess
import sys
import itertools
from pathlib import Path
from tqdm import tqdm

# Optimisation Ablation — Naïve vs Optimised Inter-Sequence

# Benchmark.cpp
# Run naive GPU OR optimised inter GPU
# Print results

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK = PROJECT_ROOT / "bin" / "benchmark"
NUM_SEQS = 600_000  # larger than Swiss-Prot so all ~574k sequences are loaded
THRESHOLD = 100_000  # high enough that every sequence is routed through inter kernel
ALGORITHMS = ["sw"]
LABELS = {"parasail": "Parasail", "gpu": "GPU Inter"}
KEYS = ["parasail", "gpu"]
QUERY_LENGTHS = [128, 256, 352, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
DATABASE_PATH = PROJECT_ROOT / "data" / "input" / "uniprot_sprot.fasta"


def run_benchmark(
    algo: str, query_len: int, database_path: str
) -> dict[str, dict[str, float]]:
    cmd = [
        str(BENCHMARK),
        algo,
        str(NUM_SEQS),
        str(THRESHOLD),
        str(PROJECT_ROOT / "data" / "input" / f"query_{query_len}.fasta"),
        database_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Benchmark failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = proc.stdout.strip().splitlines()

    # FIX 2: Check for exactly 2 lines (matching the 2 KEYS)
    if len(lines) != len(KEYS):
        print(
            f"\n[ERROR] Expected {len(KEYS)} output lines, got {len(lines)}: {proc.stdout!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    # benchmark.cpp prints: parasail, naive gpu, gpu  (in that order)
    result = {}
    for key, line in zip(KEYS, lines):
        try:
            ms, gcups = map(float, line.split())
        except ValueError:
            print(f"\n[ERROR] Could not parse line {line!r}", file=sys.stderr)
            sys.exit(1)
        result[key] = {"latency_ms": ms, "gcups": gcups}
    return result


OUTPUT_CSV = PROJECT_ROOT / "results" / "experiment_3.csv"


def write_csv(results: dict[str, dict[int, dict[str, dict[str, float]]]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Implementation", "Algorithm", "Query Length", "Latency (ms)", "GCUPS"]
        )
        for algo in ALGORITHMS:
            for query_len in QUERY_LENGTHS:
                for key in KEYS:
                    r = results[algo][query_len][key]
                    writer.writerow(
                        [
                            LABELS[key],
                            algo.upper(),
                            query_len,
                            f"{r['latency_ms']:.1f}",
                            f"{r['gcups']:.2f}",
                        ]
                    )
    print(f"Results written to {OUTPUT_CSV}")


def main() -> None:
    if not BENCHMARK.exists():
        print(f"[ERROR] Binary not found: {BENCHMARK}", file=sys.stderr)
        print("Build the project first (cmake + make).", file=sys.stderr)
        sys.exit(1)

    # Initialize the results dictionary cleanly for all algorithms upfront
    results: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        algo: {} for algo in ALGORITHMS
    }

    # Create a flat list of all (algorithm, query_length) combinations
    tasks = list(itertools.product(ALGORITHMS, QUERY_LENGTHS))

    print(f"Starting {len(tasks)} benchmark tasks. Routing up to {NUM_SEQS:,} seqs...")

    # Wrap the tasks in tqdm for the progress bar
    # Using dynamic descriptions so you can see exactly which task is running
    progress_bar = tqdm(tasks, total=len(tasks), unit="run")

    for algo, query_len in progress_bar:
        # Update the progress bar text to show current parameters
        progress_bar.set_description(f"Running {algo.upper()} (Len: {query_len})")

        # Run the actual benchmark
        results[algo][query_len] = run_benchmark(algo, query_len, str(DATABASE_PATH))

    # Clear the bar and print completion
    print("\nAll benchmarks complete!")
    write_csv(results)


if __name__ == "__main__":
    main()

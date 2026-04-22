import csv
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Optimisation Ablation — Final Hybrid vs Parasail Comparison

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK = PROJECT_ROOT / "bin" / "benchmark"

NUM_SEQS = 600_000  # Full Swiss-Prot
DATABASE_PATH = PROJECT_ROOT / "data" / "input" / "uniprot_sprot.fasta"
OUTPUT_CSV = PROJECT_ROOT / "results" / "experiment_7.csv"

# The queries from Experiment 3
QUERY_LENGTHS = [64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
ALGORITHMS = ["nw", "sw"]
KEYS = ["parasail", "gpu"]

# IMPORTANT: Update this with the optimal tau found in Experiment 6!
OPTIMAL_THRESHOLD = 8192


def run_benchmark(algo: str, query_len: int) -> dict[str, dict[str, float]]:
    """Runs the benchmark and returns latency and GCUPS for both implementations."""
    query_path = PROJECT_ROOT / "data" / "input" / f"query_{query_len}.fasta"

    if not query_path.exists():
        print(f"\n[ERROR] Query file missing: {query_path}", file=sys.stderr)
        print("Please ensure queries from Experiment 3 are present.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        str(BENCHMARK),
        algo,
        str(NUM_SEQS),
        str(OPTIMAL_THRESHOLD),
        str(query_path),
        str(DATABASE_PATH),
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


def write_csv(csv_data: list[dict]) -> None:
    """Writes the results including calculated speedups to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Algorithm",
                "Query Length",
                "Implementation",
                "Latency (ms)",
                "GCUPS",
                "Speedup vs Parasail",
            ]
        )
        for row in csv_data:
            writer.writerow(
                [
                    row["algorithm"].upper(),
                    row["query_length"],
                    row["implementation"],
                    f"{row['latency_ms']:.2f}",
                    f"{row['gcups']:.3f}",
                    f"{row['speedup']:.2f}x" if row["speedup"] else "N/A",
                ]
            )
    print(f"\nResults successfully written to {OUTPUT_CSV}")


def main() -> None:
    if not BENCHMARK.exists():
        print(f"[ERROR] Binary not found: {BENCHMARK}", file=sys.stderr)
        print("Build the project first (cmake + make).", file=sys.stderr)
        sys.exit(1)

    if not DATABASE_PATH.exists():
        print(f"[ERROR] Swiss-Prot database missing: {DATABASE_PATH}", file=sys.stderr)
        sys.exit(1)

    csv_data = []

    # Track speedups to calculate the headline figures
    speedups = {"nw": [], "sw": []}

    tasks = [(algo, length) for algo in ALGORITHMS for length in QUERY_LENGTHS]
    print(f"\nStarting Experiment 7: Final Evaluation ({len(tasks)} runs).")
    print(f"Using Optimal Threshold (tau) = {OPTIMAL_THRESHOLD}")

    progress_bar = tqdm(tasks, total=len(tasks), unit="run")

    for algo, length in progress_bar:
        progress_bar.set_description(f"{algo.upper()} | Query: {length}")

        run_res = run_benchmark(algo, length)

        # Calculate speedup
        gpu_gcups = run_res["gpu"]["gcups"]
        parasail_gcups = run_res["parasail"]["gcups"]

        # Record Parasail baseline
        csv_data.append(
            {
                "algorithm": algo,
                "query_length": length,
                "implementation": "Parasail (CPU)",
                "latency_ms": run_res["parasail"]["latency_ms"],
                "gcups": parasail_gcups,
                "speedup": None,
            }
        )

        # Record GPU Hybrid
        csv_data.append(
            {
                "algorithm": algo,
                "query_length": length,
                "implementation": "GPU Hybrid (Optimised)",
                "latency_ms": run_res["gpu"]["latency_ms"],
                "gcups": gpu_gcups,
                "speedup": None,
            }
        )

    write_csv(csv_data)


if __name__ == "__main__":
    main()

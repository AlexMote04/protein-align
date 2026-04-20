import csv
import subprocess
import sys
from pathlib import Path

# Optimisation Ablation — Naïve vs Optimised Inter-Sequence

# Benchmark.cpp
# Run naive GPU OR optimised inter GPU
# Print results

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK = PROJECT_ROOT / "bin" / "benchmark"
NUM_SEQS = 600_000  # larger than Swiss-Prot so all ~574k sequences are loaded
THRESHOLD = 100_000  # high enough that every sequence is routed through inter kernel
ALGORITHMS = ["nw", "sw"]
LABELS = {"parasail": "Parasail", "naive gpu": "Naive GPU Inter", "gpu": "GPU Inter"}
KEYS = ["parasail", "naive gpu", "gpu"]


def run_benchmark(algo: str) -> dict[str, dict[str, float]]:
    cmd = [str(BENCHMARK), algo, str(NUM_SEQS), str(THRESHOLD)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Benchmark failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = proc.stdout.strip().splitlines()
    if len(lines) < 3:
        print(
            f"\n[ERROR] Expected 3 output lines, got: {proc.stdout!r}", file=sys.stderr
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


OUTPUT_CSV = PROJECT_ROOT / "results" / "experiment_2.csv"


def write_csv(results: dict[str, dict[str, dict[str, float]]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Implementation", "Algorithm", "Latency (ms)", "GCUPS"])
        for algo in ALGORITHMS:
            for key in KEYS:
                r = results[algo][key]
                writer.writerow(
                    [
                        LABELS[key],
                        algo.upper(),
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

    results: dict[str, dict[str, dict[str, float]]] = {}
    for algo in ALGORITHMS:
        print(
            f"Running {algo.upper()} (threshold={THRESHOLD:,}, up to {NUM_SEQS:,} seqs)...",
            flush=True,
        )
        results[algo] = run_benchmark(algo)
        print("  Done.")

    write_csv(results)


if __name__ == "__main__":
    main()

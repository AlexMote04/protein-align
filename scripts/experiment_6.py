import csv
import subprocess
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Optimisation Ablation — Hybrid Threshold Tuning

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK = PROJECT_ROOT / "bin" / "benchmark"

NUM_SEQS = 600_000  # Large enough to cover the full Swiss-Prot dataset
DATABASE_PATH = PROJECT_ROOT / "data" / "input" / "uniprot_sprot.fasta"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_CSV = PROJECT_ROOT / "results" / "experiment_6.csv"

# Query lengths representing short, medium, and long sequences
QUERY_LENGTHS = [128, 464, 2048]

# Thresholds to sweep.
# 1 = Pure Intra, 1_000_000 = Pure Inter
THRESHOLDS = [512, 1024, 2048, 4096, 8192, 12288, 16384, 1_000_000]

KEYS = ["parasail", "gpu"]


def generate_query_fasta(path: Path, length: int) -> None:
    """Generates a random query FASTA if it doesn't already exist."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    with path.open("w") as f:
        seq = "".join(random.choices(amino_acids, k=length))
        f.write(f">query_len_{length}\n{seq}\n")


def prepare_queries() -> dict[int, Path]:
    """Ensures query files exist for the specified lengths."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    query_paths = {}
    for length in QUERY_LENGTHS:
        q_path = INPUT_DIR / f"query_{length}.fasta"
        if not q_path.exists():
            print(f"Generating query of length {length}...")
            generate_query_fasta(q_path, length)
        query_paths[length] = q_path
    return query_paths


def run_benchmark(
    algo: str, query_path: Path, threshold: int
) -> dict[str, dict[str, float]]:
    """Executes the benchmark binary and parses latency and GCUPS."""
    cmd = [
        str(BENCHMARK),
        algo,
        str(NUM_SEQS),
        str(threshold),
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


def write_csv(results: list[dict]) -> None:
    """Writes the flat list of results to a CSV file."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Algorithm",
                "Query Length",
                "Threshold",
                "Implementation",
                "Latency (ms)",
                "GCUPS",
            ]
        )
        for row in results:
            # Map friendly names for special thresholds
            th_label = row["threshold"]
            if th_label == 0:
                th_label = "0 (Pure Intra)"
            elif th_label == 1_000_000:
                th_label = "Pure Inter"

            writer.writerow(
                [
                    row["algorithm"].upper(),
                    row["query_length"],
                    th_label,
                    row["implementation"],
                    f"{row['latency_ms']:.2f}",
                    f"{row['gcups']:.3f}",
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

    query_paths = prepare_queries()
    csv_data = []

    # --- Phase 1: Full Sweep for SW ---
    tasks = [(length, th) for length in QUERY_LENGTHS for th in THRESHOLDS]
    print(f"\nStarting Phase 1: SW Threshold Sweep ({len(tasks)} runs)...")

    progress_bar = tqdm(tasks, total=len(tasks), unit="run")
    sw_results = {
        length: {} for length in QUERY_LENGTHS
    }  # To help find the optimal later

    for length, threshold in progress_bar:
        th_display = (
            "Pure Intra"
            if threshold == 0
            else "Pure Inter" if threshold == 1_000_000 else str(threshold)
        )
        progress_bar.set_description(f"SW | Query: {length} | Thresh: {th_display}")

        run_res = run_benchmark("sw", query_paths[length], threshold)

        # Store for optimal calculation
        sw_results[length][threshold] = run_res["gpu"]["latency_ms"]

        csv_data.append(
            {
                "algorithm": "sw",
                "query_length": length,
                "threshold": threshold,
                "implementation": "GPU Hybrid",
                "latency_ms": run_res["gpu"]["latency_ms"],
                "gcups": run_res["gpu"]["gcups"],
            }
        )

        # Only record Parasail once per query length to avoid CSV clutter
        if threshold == 1_000_000:
            csv_data.append(
                {
                    "algorithm": "sw",
                    "query_length": length,
                    "threshold": "N/A",
                    "implementation": "Parasail",
                    "latency_ms": run_res["parasail"]["latency_ms"],
                    "gcups": run_res["parasail"]["gcups"],
                }
            )

    # --- Phase 2: Find Optimal Threshold and run NW ---
    # We use the mid-length query (464) to determine the best threshold,
    # ignoring the "pure" bounds to ensure a true hybrid threshold.
    mid_query_results = sw_results[464]
    valid_thresholds = [t for t in THRESHOLDS if t not in (0, 1_000_000)]

    best_threshold = min(valid_thresholds, key=lambda t: mid_query_results[t])
    print(f"\nOptimal threshold found for SW (query=500): {best_threshold}")
    print(f"Starting Phase 2: Running NW at optimal threshold ({best_threshold})...")

    for length in QUERY_LENGTHS:
        run_res = run_benchmark("nw", query_paths[length], best_threshold)

        csv_data.append(
            {
                "algorithm": "nw",
                "query_length": length,
                "threshold": best_threshold,
                "implementation": "GPU Hybrid",
                "latency_ms": run_res["gpu"]["latency_ms"],
                "gcups": run_res["gpu"]["gcups"],
            }
        )

    write_csv(csv_data)


if __name__ == "__main__":
    main()

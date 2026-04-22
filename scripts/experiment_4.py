import csv
import subprocess
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Optimisation Ablation — Database Composition Sensitivity (Inter-Sequence)

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK = PROJECT_ROOT / "bin" / "benchmark"

NUM_SEQS = 100_000
THRESHOLD = 1_000_000  # Extremely high to ensure all 100k sequences route through the inter kernel
ALGORITHMS = ["sw"]
LABELS = {"parasail": "Parasail", "gpu": "GPU Inter"}
KEYS = ["parasail", "gpu"]

# Use a single mid-length query (~360 residues).
# Using 352 here assuming it already exists from Experiment 3.
QUERY_LENGTH = 464
INPUT_DIR = PROJECT_ROOT / "data" / "input"
QUERY_FILE = INPUT_DIR / f"query_{QUERY_LENGTH}.fasta"

SWISSPROT_FULL = INPUT_DIR / "uniprot_sprot.fasta"
OUTPUT_CSV = PROJECT_ROOT / "results" / "experiment_4.csv"

DATABASES = {
    "Uniform 8": INPUT_DIR / "exp4_uniform_8.fasta",
    "Uniform 16": INPUT_DIR / "exp4_uniform_16.fasta",
    "Uniform 32": INPUT_DIR / "exp4_uniform_32.fasta",
    "Uniform 64": INPUT_DIR / "exp4_uniform_64.fasta",
    "Uniform 128": INPUT_DIR / "exp4_uniform_128.fasta",
    "Uniform 256": INPUT_DIR / "exp4_uniform_256.fasta",
    "Uniform 512": INPUT_DIR / "exp4_uniform_512.fasta",
    "Uniform 768": INPUT_DIR / "exp4_uniform_768.fasta",
    "Uniform 1024": INPUT_DIR / "exp4_uniform_1024.fasta",
    "Swiss-Prot": INPUT_DIR / "sprot_100k_subset.fasta",
}


def generate_random_fasta(path: Path, lengths: list[int]) -> None:
    """Generates a FASTA file with random sequences of the specified lengths."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    with path.open("w") as f:
        for i, l in enumerate(lengths):
            seq = "".join(random.choices(amino_acids, k=l))
            f.write(f">seq_{i}_len_{l}\n{seq}\n")


def prepare_databases() -> None:
    """Generates synthetic datasets and the sorted Swiss-Prot subset if they don't exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATABASES["Uniform 8"].exists():
        print("Generating Uniform 8 database...")
        generate_random_fasta(DATABASES["Uniform 8"], [8] * NUM_SEQS)

    if not DATABASES["Uniform 16"].exists():
        print("Generating Uniform 16 database...")
        generate_random_fasta(DATABASES["Uniform 16"], [16] * NUM_SEQS)

    if not DATABASES["Uniform 32"].exists():
        print("Generating Uniform 32 database...")
        generate_random_fasta(DATABASES["Uniform 32"], [32] * NUM_SEQS)

    if not DATABASES["Uniform 64"].exists():
        print("Generating Uniform 64 database...")
        generate_random_fasta(DATABASES["Uniform 64"], [64] * NUM_SEQS)

    if not DATABASES["Uniform 128"].exists():
        print("Generating Uniform 128 database...")
        generate_random_fasta(DATABASES["Uniform 128"], [128] * NUM_SEQS)

    if not DATABASES["Uniform 256"].exists():
        print("Generating Uniform 256 database...")
        generate_random_fasta(DATABASES["Uniform 256"], [256] * NUM_SEQS)

    if not DATABASES["Uniform 512"].exists():
        print("Generating Uniform 512 database...")
        generate_random_fasta(DATABASES["Uniform 512"], [512] * NUM_SEQS)

    if not DATABASES["Uniform 768"].exists():
        print("Generating Uniform 768 database...")
        generate_random_fasta(DATABASES["Uniform 768"], [768] * NUM_SEQS)

    if not DATABASES["Uniform 1024"].exists():
        print("Generating Uniform 1024 database...")
        generate_random_fasta(DATABASES["Uniform 1024"], [1024] * NUM_SEQS)


def run_benchmark(algo: str, database_path: Path) -> dict[str, dict[str, float]]:
    cmd = [
        str(BENCHMARK),
        algo,
        str(NUM_SEQS),
        str(THRESHOLD),
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


def write_csv(results: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Dataset", "Implementation", "Algorithm", "Latency (ms)", "GCUPS"]
        )
        for algo in ALGORITHMS:
            for db_name in DATABASES:
                for key in KEYS:
                    r = results[algo][db_name][key]
                    writer.writerow(
                        [
                            db_name,
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

    if not QUERY_FILE.exists():
        print(f"[ERROR] Query file not found: {QUERY_FILE}", file=sys.stderr)
        print(
            "Ensure you have generated the query file from the previous experiment.",
            file=sys.stderr,
        )
        sys.exit(1)

    prepare_databases()

    # Dictionary: results[algo][db_name] = { "parasail": {...}, "gpu": {...} }
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        algo: {} for algo in ALGORITHMS
    }

    # Create a flat list of (algorithm, db_name)
    tasks = [(algo, db_name) for algo in ALGORITHMS for db_name in DATABASES]

    print(f"\nStarting {len(tasks)} benchmark tasks using query length {QUERY_LENGTH}.")

    progress_bar = tqdm(tasks, total=len(tasks), unit="run")

    for algo, db_name in progress_bar:
        progress_bar.set_description(f"Running {algo.upper()} on {db_name}")
        results[algo][db_name] = run_benchmark(algo, DATABASES[db_name])

    print("\nAll benchmarks complete!")
    write_csv(results)


if __name__ == "__main__":
    main()

import random
import argparse
import sys
from pathlib import Path


def extract_random_subset(
    input_file: Path, output_file: Path, num_sequences: int
) -> None:
    """Parses a FASTA file, samples sequences without replacement, and writes them out."""
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {input_file.name}...")

    # 1. Parse the FASTA file into memory
    sequences = []
    current_header = ""
    current_seq = []

    with input_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save the previous sequence before starting a new one
                if current_header:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)

        # Catch the final sequence at the end of the file
        if current_header:
            sequences.append((current_header, "".join(current_seq)))

    total_seqs = len(sequences)
    print(f"Successfully loaded {total_seqs:,} sequences.")

    # 2. Validate sample size
    if num_sequences > total_seqs:
        print(
            f"[WARNING] Requested {num_sequences:,} sequences, but only {total_seqs:,} exist. Exporting all.",
            file=sys.stderr,
        )
        num_sequences = total_seqs

    # 3. Sample without replacement
    print(f"Randomly sampling {num_sequences:,} sequences without replacement...")
    sampled_sequences = random.sample(sequences, num_sequences)

    # 4. Write out the new FASTA
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for header, seq in sampled_sequences:
            # We fold the sequence to a standard 80 character width for neatness,
            # though your C++ parser likely handles single-line sequences fine too.
            f.write(f"{header}\n")
            # Write sequence in chunks of 80 characters
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")

    print(f"Subset successfully written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly sample sequences from a FASTA file."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Path to the input FASTA file."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Path for the output FASTA file.",
    )
    parser.add_argument(
        "-n", "--num", required=True, type=int, help="Number of sequences to sample."
    )

    args = parser.parse_args()
    extract_random_subset(args.input, args.output, args.num)

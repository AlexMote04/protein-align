import subprocess
import matplotlib.pyplot as plt
import argparse
import sys


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Run sequence alignment benchmarks and plot results."
    )
    parser.add_argument(
        "algorithm", choices=["nw", "sw"], help="The algorithm to run ('nw' or 'sw')"
    )
    args = parser.parse_args()

    algo = args.algorithm

    # Generate the sequence numbers (160 doubling up to 81920)
    num_seqs_list = []
    n = 160
    while n <= 81920:
        num_seqs_list.append(n)
        n *= 2

    # Data structures to hold the parsed metrics
    metrics = {
        "cpu_ms": [],
        "parasail_ms": [],
        "gpu_ms": [],
        "cpu_gcups": [],
        "parasail_gcups": [],
        "gpu_gcups": [],
        "cpu_gpu_speedup": [],
        "parasail_gpu_speedup": [],
    }

    # Execution Loop
    print(f"Starting benchmark for algorithm: {algo.upper()}")
    print("-" * 50)

    for seq in num_seqs_list:
        print(f"Running {seq} sequences...", end="", flush=True)

        # Construct the command
        cmd = ["./benchmark", algo, str(seq)]

        try:
            # Run the C++ program and capture the standard output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()

            # Parse the comma-separated floating point numbers
            vals = [float(x) for x in output.split(",")]

            # Map values to our dictionary (Order must match your C++ output exactly)
            metrics["cpu_ms"].append(vals[0])
            metrics["parasail_ms"].append(vals[1])
            metrics["gpu_ms"].append(vals[2])
            metrics["cpu_gcups"].append(vals[3])
            metrics["parasail_gcups"].append(vals[4])
            metrics["gpu_gcups"].append(vals[5])
            metrics["cpu_gpu_speedup"].append(vals[6])
            metrics["parasail_gpu_speedup"].append(vals[7])

            print(" Done.")

        except subprocess.CalledProcessError as e:
            print(
                f"\n[ERROR] The C++ program crashed or returned an error for {seq} sequences."
            )
            print(f"Error details: {e.stderr}")
            sys.exit(1)
        except ValueError:
            print(
                f"\n[ERROR] Failed to parse output. Expected 8 comma-separated floats."
            )
            print(f"Raw output received: '{output}'")
            sys.exit(1)
        except IndexError:
            print(f"\n[ERROR] Missing data in output. Expected 8 values.")
            print(f"Raw output received: '{output}'")
            sys.exit(1)

    print("-" * 50)
    print("Benchmarking complete. Generating plots...")

    # Plotting
    # Note: We use a Log2 scale for the X-axis because sequence sizes double each step.
    # This prevents the smaller data points from getting squished together on the left side.

    # Plot A: Latency (Log-Log scale is usually best for time vs doubling workload)
    plt.figure(figsize=(10, 6))
    plt.plot(num_seqs_list, metrics["cpu_ms"], marker="o", label="CPU")
    plt.plot(num_seqs_list, metrics["parasail_ms"], marker="s", label="Parasail")
    plt.plot(num_seqs_list, metrics["gpu_ms"], marker="^", label="GPU")

    plt.xscale("log", base=2)
    plt.yscale("log")  # Log scale for Y because time usually grows exponentially
    plt.xlabel("Number of Sequences")
    plt.ylabel("Latency (ms) - Log Scale")
    plt.title(f"{algo.upper()} Latency vs. Sequence Count")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{algo}_latency_plot.png", dpi=300)
    plt.close()

    # Plot B: Throughput
    plt.figure(figsize=(10, 6))
    plt.plot(num_seqs_list, metrics["cpu_gcups"], marker="o", label="CPU")
    plt.plot(num_seqs_list, metrics["parasail_gcups"], marker="s", label="Parasail")
    plt.plot(num_seqs_list, metrics["gpu_gcups"], marker="^", label="GPU")

    plt.xscale("log", base=2)
    plt.xlabel("Number of Sequences")
    plt.ylabel("Throughput (GCUPS)")
    plt.title(f"{algo.upper()} Throughput vs. Sequence Count")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{algo}_throughput_plot.png", dpi=300)
    plt.close()

    # Plot C: Speedups
    plt.figure(figsize=(10, 6))
    plt.plot(
        num_seqs_list,
        metrics["cpu_gpu_speedup"],
        marker="o",
        label="GPU vs CPU Speedup",
    )
    plt.plot(
        num_seqs_list,
        metrics["parasail_gpu_speedup"],
        marker="s",
        label="GPU vs Parasail Speedup",
    )

    # Add a red dashed line at y=1 to easily see GPU becomes faster
    plt.axhline(y=1, color="r", linestyle="--", label="1x (Break-even)")

    plt.xscale("log", base=2)
    plt.xlabel("Number of Sequences")
    plt.ylabel("Speedup Factor (x)")
    plt.title(f"{algo.upper()} GPU Speedup vs. Sequence Count")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{algo}_speedup_plot.png", dpi=300)
    plt.close()

    print(f"Success! Saved 3 plots to the current directory: ")
    print(f" - {algo}_latency_plot.png")
    print(f" - {algo}_throughput_plot.png")
    print(f" - {algo}_speedup_plot.png")


if __name__ == "__main__":
    main()

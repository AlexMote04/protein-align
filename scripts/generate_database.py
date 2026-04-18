import random

CHARS = "ARNDCQEGHILKMFPSTWYVBZX*"

DB_PATH = "./data/input/query_360.fasta"
NUM_SEQUENCES = 1  # Number of sequences in database
SEQUENCE_LEN = 360  # Length of each target sequence

with open(DB_PATH, "w", encoding="utf-8") as outfile:
    for sequence in range(NUM_SEQUENCES):
        outfile.write(">Example Sequence\n")
        outstring = "".join(random.choice(CHARS) for _ in range(SEQUENCE_LEN))
        outstring += "\n"
        outfile.write(outstring)

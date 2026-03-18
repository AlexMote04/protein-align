import random

CHARS = "ARNDCQEGHILKMFPSTWYVBZX*"

DB_PATH = "./data/input/fixed_len.fasta"
NUM_SEQUENCES = 131072  # Number of sequences in database
SEQUENCE_LEN = 256  # Number of db sequences

with open(DB_PATH, "w", encoding="utf-8") as outfile:
    for sequence in range(NUM_SEQUENCES):
        outfile.write(">Example Sequence\n")
        outstring = "".join(random.choice(CHARS) for _ in range(SEQUENCE_LEN))
        outstring += "\n"
        outfile.write(outstring)

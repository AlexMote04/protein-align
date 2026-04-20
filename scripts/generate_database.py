import random

CHARS = "ARNDCQEGHILKMFPSTWYVBZX*"

SEQ_LENS = [128, 1024]

DB_PATH = f"./data/input/db_bimodal.fasta"
NUM_SEQUENCES = 100_000  # Number of sequences in database
# SEQUENCE_LEN = 352  # Length of each target sequence

with open(DB_PATH, "w", encoding="utf-8") as outfile:
    for sequence in range(NUM_SEQUENCES):
        outfile.write(">Example Sequence\n")
        outstring = "".join(random.choice(CHARS) for _ in range(SEQ_LENS[sequence % 2]))
        outstring += "\n"
        outfile.write(outstring)

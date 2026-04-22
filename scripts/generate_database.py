import random

CHARS = "ARNDCQEGHILKMFPSTWYVBZX*"
NUM_SEQUENCES = 1  # Number of sequences in database
SEQ_LEN = 464
DB_PATH = f"./data/input/query_464.fasta"

# SEQUENCE_LEN = 352  # Length of each target sequence
with open(DB_PATH, "w", encoding="utf-8") as outfile:
    for sequence in range(NUM_SEQUENCES):
        outfile.write(">Example Sequence\n")
        outstring = "".join(random.choice(CHARS) for _ in range(SEQ_LEN))
        outstring += "\n"
        outfile.write(outstring)

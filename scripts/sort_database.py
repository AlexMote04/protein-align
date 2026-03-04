input_path = "./data/input/uniprot_sprot"
output_path = "./data/input/uniprot_sprot_sorted"

sequences = []
with open(input_path, "r", encoding="utf-8") as file:
    current_seq = []

    for line in file:
        line = line.strip()
        if not line:
            continue  # Skip emtpy lines

        if line.startswith(">"):
            if current_seq:
                sequences.append("".join(current_seq))
            current_seq = []
        else:
            current_seq.append(line)

    if current_seq:
        sequences.append("".join(current_seq))

sequences.sort(key=len)

with open(output_path, "x", encoding="utf-8") as file:
    for sequence in sequences:
        file.write(f">\n{sequence}\n")

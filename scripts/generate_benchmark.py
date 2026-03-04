import random
CHARS = "ARNDCQEGHILKMFPSTWYVBZX*"

QUERY_OUT = "./data/input/query"
DB_OUT = "./data/input/db"
N = 4 # Number of output dbs
L = 100 # Number of db sequences


# Generate query
query = ">test\n"
for char in range(512):
  query += CHARS[random.randint(0, len(CHARS) - 1)]

with open(QUERY_OUT, "x", encoding='utf-8') as qfile:
  qfile.write(query)

for n in range(1):
  db = ""
  for seq in range(L):
    db += ">test\n"
    for char in range(512):
      db += CHARS[random.randint(0, len(CHARS) - 1)]
    db += "\n"

  with open(DB_OUT + str(n*100) + "k", "x", encoding='utf-8') as dbfile:
    dbfile.write(db)

  N *= 2
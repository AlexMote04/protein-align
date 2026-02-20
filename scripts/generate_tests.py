import random
CHARS = "ARNDCQEGHILKMFPSTWYVBZX*"

QUERY_OUT = "./data/test/auto/q"
DB_OUT = "./data/test/auto/db"
N = 5 # Number of query-db pairs to generate
M = 12 # Number of sequences to add to database
L = 100 # Max sequence length

for n in range(N):
  # Generate query
  query = ">test\n"
  for l in range(random.randint(1, L)):
    query += CHARS[random.randint(0, len(CHARS) - 1)]

  with open(QUERY_OUT + str(n), "x", encoding='utf-8') as qfile:
    qfile.write(query)
    
  db = ""
  for m in range(M):
    db += ">test\n"
    for l in range(random.randint(1, L)):
      db += CHARS[random.randint(0, len(CHARS) - 1)]
    db += "\n"

  with open(DB_OUT + str(n), "x", encoding='utf-8') as dbfile:
    dbfile.write(db)
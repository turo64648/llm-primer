# Run: ~/.venvs/llm/bin/python exercises/ch11_rag.py
# A tiny retrieval system (TF-IDF + cosine top-k), three ways it fails, and a p^n compounding table.
import re
import numpy as np

np.random.seed(0)

# A small "company wiki". Each string is one document.
CORPUS = [
    "To rotate TLS certificates, run certctl rotate on the gateway and restart the proxy.",
    "The billing service writes invoices to the ledger database every night at 02:00.",
    "Deploys go through the staging cluster first; promote with deployctl promote.",
    "The March 4 incident: an expired certificate on the gateway made every request fail for 40 minutes.",
    "Kafka consumers in the ingest pipeline commit offsets every 5 seconds.",
    "On-call engineers page the database team through the dbops rotation.",
    "The search index is rebuilt every Sunday from the product catalog.",
    "Grafana dashboards for the gateway show request rate, errors and latency.",
    "Backups of the ledger database are kept for 35 days in cold storage.",
    "The mobile app talks to the gateway over HTTP/2 with retries and backoff.",
]

STOP = set("a an and the to of in on for with is are be by at from every first over made".split())

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]

class TfidfIndex:
    def __init__(self, docs):
        self.docs = docs
        self.vocab = sorted({w for d in docs for w in tokenize(d)})
        self.col = {w: i for i, w in enumerate(self.vocab)}
        tf = np.array([self._counts(d) for d in docs])
        df = (tf > 0).sum(axis=0)
        # Rare words get high weight; words in every document get low weight.
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        self.mat = self._normalize(tf * self.idf)

    def _counts(self, text):
        v = np.zeros(len(self.vocab))
        for w in tokenize(text):
            if w in self.col:          # words never seen in the corpus are dropped
                v[self.col[w]] += 1
        return v

    @staticmethod
    def _normalize(m):
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def search(self, query, k=3):
        q = self._normalize(self._counts(query) * self.idf)
        scores = self.mat @ q          # cosine similarity, since every row has length 1
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.docs[i]) for i in top]

def show(index, query, k=3):
    print(f"\nQuery: {query!r}")
    known = [w for w in tokenize(query) if w in index.col]
    print(f"  query words the index knows: {known}")
    for s, d in index.search(query, k):
        print(f"  {s:.2f}  {d[:78]}")

index = TfidfIndex(CORPUS)
print(f"Indexed {len(CORPUS)} documents, vocabulary of {len(index.vocab)} words.")

print("\n=== 1. Works: the query shares words with the answer ===")
show(index, "how do I rotate the TLS certificates?")

print("\n=== 2. Fails: same fact, no shared words (vocabulary mismatch) ===")
show(index, "why was the website down last spring?")

print("\n=== 3. Fails: you did not know what to ask for ===")
# The user asks about the mobile app being slow. The real cause is the gateway
# certificate incident, but nothing in the question points there.
show(index, "why did the mobile app fail for everyone in March?")

print("\n=== 4. Chunk boundaries: a fact split across two chunks ===")
long_doc = ("The ledger database runs on host db7. Its primary replica is in zone b. "
            "Credentials for it live in vault under secret/ledger/admin. Rotate them quarterly.")
words = long_doc.split()
chunks = [" ".join(words[i:i + 12]) for i in range(0, len(words), 12)]
for i, c in enumerate(chunks):
    print(f"  chunk {i}: {c}")
cidx = TfidfIndex(chunks)
show(cidx, "where are the ledger database credentials?", k=2)
print("  -> chunk 1 holds the vault path but only says 'it'; chunk 0 says 'ledger database'.")

print("\n=== 5. Error compounding: P(all n steps succeed) = p^n ===")
ns = [1, 10, 20, 50, 100, 500]
ps = [0.90, 0.95, 0.99, 0.999]
print("  p \\ n " + "".join(f"{n:>8}" for n in ns))
for p in ps:
    print(f"  {p:<6}" + "".join(f"{p**n:>8.3f}" for n in ns))
for p in ps:
    print(f"  p={p}: 50% of runs survive about {np.log(0.5)/np.log(p):.0f} steps")

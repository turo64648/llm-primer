# Run: ~/.venvs/llm/bin/python exercises/ch03_bpe.py
"""Chapter 3: a byte-level BPE tokenizer from scratch, trained on Shakespeare.

Steps: load the text (downloaded once and cached in exercises/data/), split it into chunks with a
pre-tokenization regex, start from the 256 byte values, and repeatedly merge the most frequent
adjacent pair into a new token. Then encode and decode, and check the round trip.

Chapter 4 reuses this file:  from ch03_bpe import load_text, get_tokenizer
"""
import json
import os
import re
import time
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TEXT_PATH = os.path.join(DATA, "tinyshakespeare.txt")
TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
MERGES_PATH = os.path.join(DATA, "bpe_merges.json")
N_MERGES = 512  # vocabulary = 256 bytes + 512 merges + 1 special token = 769

# Pre-tokenization: a simplified version of GPT-2's regex. Chunks are: a word with an optional leading
# space, a single digit, a run of punctuation, or whitespace. Merges never cross chunk boundaries,
# so no token can glue the end of one word to the start of the next.
SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
EOT = "<|endoftext|>"  # a special token: never produced from text, only inserted on purpose


def load_text():
    """All of Tiny Shakespeare (about 1.1 MB). Downloads it on the first call only."""
    if not os.path.exists(TEXT_PATH):
        os.makedirs(DATA, exist_ok=True)
        print(f"downloading {TEXT_URL}")
        urllib.request.urlretrieve(TEXT_URL, TEXT_PATH)
    with open(TEXT_PATH, encoding="utf-8") as f:
        return f.read()


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in the list `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text, n_merges, verbose=True):
    """Learn `n_merges` merge rules. Returns a list of (left_id, right_id); rule k makes token 256 + k."""
    # Count each distinct chunk once, with its frequency: far fewer than the raw chunks.
    chunk_freq = Counter(SPLIT.findall(text))
    words = [(list(c.encode("utf-8")), f) for c, f in chunk_freq.items()]
    merges = []
    for k in range(n_merges):
        pairs = Counter()
        for ids, f in words:
            for a, b in zip(ids, ids[1:]):
                pairs[a, b] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # the most frequent adjacent pair
        new_id = 256 + k
        words = [(merge(ids, best, new_id), f) for ids, f in words]
        merges.append(best)
        if verbose and (k < 12 or k % 100 == 99):
            vocab = build_vocab(merges)
            print(f"merge {k + 1:3d}: {show(vocab[best[0]])} + {show(vocab[best[1]])}"
                  f" -> {show(vocab[new_id])}  (seen {pairs[best]} times)")
    return merges


def build_vocab(merges):
    """Token id -> the bytes it stands for."""
    vocab = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    return vocab


def show(b):
    """Printable form of a token's bytes."""
    return repr(b.decode("utf-8", errors="replace"))


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: 256 + k for k, pair in enumerate(self.merges)}  # earlier merge = applied first
        self.vocab = build_vocab(self.merges)
        self.eot_id = 256 + len(self.merges)
        self.vocab_size = self.eot_id + 1
        self.cache = {}

    def _encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge that is present, exactly as training did.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = merge(ids, pair, self.rank[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for i, part in enumerate(text.split(EOT)):  # special tokens are cut out before BPE runs
            if i > 0:
                out.append(self.eot_id)
            for chunk in SPLIT.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        b = b"".join(EOT.encode() if i == self.eot_id else self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")  # a cut in the middle of a character shows as �


def get_tokenizer(verbose=False):
    """Load the merges from exercises/data/bpe_merges.json, training them first if needed."""
    if not os.path.exists(MERGES_PATH):
        merges = train_bpe(load_text(), N_MERGES, verbose=verbose)
        with open(MERGES_PATH, "w") as f:
            json.dump(merges, f)
    with open(MERGES_PATH) as f:
        return Tokenizer(json.load(f))


if __name__ == "__main__":
    text = load_text()
    print(f"text: {len(text):,} characters, {len(text.encode('utf-8')):,} bytes")
    print("first chunks:", SPLIT.findall(text[:60]))

    print(f"\n== Training {N_MERGES} merges ==")
    t0 = time.time()
    merges = train_bpe(text, N_MERGES)
    print(f"trained in {time.time() - t0:.0f} s")
    with open(MERGES_PATH, "w") as f:
        json.dump(merges, f)

    tok = Tokenizer(merges)
    longest = sorted(tok.vocab.values(), key=len, reverse=True)[:8]
    print(f"\nvocabulary size: {tok.vocab_size}  (256 bytes + {len(merges)} merges + 1 special)")
    print("longest tokens:", ", ".join(show(b) for b in longest))

    print("\n== Encoding ==")
    ids = tok.encode(text)
    print(f"whole text: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens"
          f"  ({len(text.encode('utf-8')) / len(ids):.2f} bytes per token)")
    for s in ["To be, or not to be: that is the question.",
              "The king's horse ran 1234 miles.",
              "Shakespeare, naïve café 日本"]:
        ids = tok.encode(s)
        pieces = [tok.vocab[i].decode("utf-8", errors="replace") for i in ids]
        print(f"\n{s!r}\n  {len(s.encode('utf-8'))} bytes -> {len(ids)} tokens: {pieces}")
        assert tok.decode(ids) == s
    ids = tok.encode("Exeunt." + EOT + "ACT II")
    print(f"\nwith a special token: {ids}  -> {tok.decode(ids)!r}")
    print("round trip OK: decode(encode(s)) == s for every example")

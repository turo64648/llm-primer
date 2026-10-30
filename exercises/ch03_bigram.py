# Run: ~/.venvs/llm/bin/python exercises/ch03_bigram.py
"""Chapter 3: the smallest language model. A bigram model predicts the next token from the current
token only. Its whole network is one embedding table of shape (vocab, vocab): row t holds the logits
for "what comes after token t". Trained by gradient descent on Shakespeare, first on raw bytes, then
on the BPE tokens from ch03_bpe.py.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch03_bpe import get_tokenizer, load_text  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)


class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)  # row = logits for the next token

    def forward(self, idx):          # idx: (batch,) token ids
        return self.table(idx)       # (batch, vocab) logits: a row lookup, nothing else


def train(name, ids, vocab_size, decode, steps=1500, batch=4096):
    data = torch.tensor(ids)
    n = int(0.9 * len(data))
    train_ids, val_ids = data[:n], data[n:]          # last 10% held out
    model = Bigram(vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)

    def val_loss():
        with torch.no_grad():
            return F.cross_entropy(model(val_ids[:-1]), val_ids[1:]).item()

    print(f"\n== {name}: vocab {vocab_size}, {len(data):,} tokens,"
          f" {vocab_size * vocab_size:,} parameters ==")
    print(f"guessing uniformly: loss = ln({vocab_size}) = {math.log(vocab_size):.2f}")
    for step in range(steps + 1):
        i = torch.randint(0, n - 1, (batch,))
        x, y = train_ids[i], train_ids[i + 1]       # (current token, the token that follows)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0:
            print(f"step {step:4d}  train loss {loss.item():.3f}  val loss {val_loss():.3f}")

    vl = val_loss()
    bytes_per_token = len(decode(val_ids.tolist()).encode("utf-8")) / len(val_ids)
    print(f"val perplexity {math.exp(vl):.1f}   val loss per byte {vl / bytes_per_token:.3f} nats")

    # Generate: start from a newline, sample from softmax(logits) again and again.
    idx = torch.tensor([ids[0] if name.startswith("BPE") else 10])
    out = [idx.item()]
    with torch.no_grad():
        for _ in range(60 if name.startswith("BPE") else 150):
            probs = F.softmax(model(idx), dim=-1)
            idx = torch.multinomial(probs, 1)[:, 0]
            out.append(idx.item())
    print("sample:\n" + decode(out))
    return model


text = load_text()

# Part 1: bytes. The text is ASCII, so one byte = one character; vocab = 256.
byte_ids = list(text.encode("utf-8"))
train("Bytes", byte_ids, 256, lambda ids: bytes(ids).decode("utf-8", errors="replace"))

# Part 2: BPE tokens from ch03_bpe.py (trains the merges first if exercises/data has none).
tok = get_tokenizer()
model = train("BPE tokens", tok.encode(text), tok.vocab_size, tok.decode)

# Part 3: geometry. Rows of the table are next-token predictions; tokens that are followed by
# similar things end up with similar rows. Nearest neighbours by cosine similarity:
W = F.normalize(model.table.weight.detach(), dim=1)
print("\n== Nearest rows (cosine similarity) ==")
for word in [" he", " my", " the", " king", " love"]:
    ids = tok.encode(word)
    if len(ids) != 1:
        continue
    sims = W @ W[ids[0]]
    best = sims.topk(6).indices.tolist()[1:]       # skip the token itself
    print(f"{word!r:9} -> " + ", ".join(repr(tok.decode([j])) for j in best))

---
title: "3. Text as Numbers"
---

<script setup>
import { cards } from './tokens-embeddings-review'
</script>

# 3. Text as Numbers

A neural network computes with numbers, so text must become numbers before a model can read it, and numbers
must become text again on the way out. Interviewers ask about this step because it explains real costs and
real failures: per-token pricing, weak spelling and arithmetic, and higher bills for non-English users.

::: info Before you start
- A neural network is layers of the form h = f(Wx + b): a matrix multiply, a bias and a simple bend. Its
  numbers are its parameters ([Chapter 1](/foundations/neural-nets)).
- Training measures how wrong the network is with a loss, computes the gradient by backpropagation, and
  nudges every parameter downhill ([Chapter 2](/foundations/learning)).
- A classifier outputs one raw score (logit) per option; softmax turns the scores into probabilities, and the
  loss is −log of the probability given to the right answer ([Chapter 2](/foundations/learning)).

The chapter explains what it needs as it goes.
:::

## The problem: text is not a vector

**In short:** a model needs a fixed list of symbols, each with an id number. The question is what one symbol
should be: a character, a word, or something in between.

A network's first layer multiplies a vector by a matrix. The string "The cat sat" is not a vector. So the
first job is to cut text into pieces from a fixed list, and give each piece an integer id. The model sees
only the ids.

Each piece is a <Term id="token">token</Term>. The fixed list of all tokens is the <Term id="vocabulary">vocabulary</Term>, and the program that
cuts text into tokens and back is the <Term id="tokenizer">tokenizer</Term>. The tokenizer is built once, before the model is trained, and
never changes afterwards. There are two obvious choices for a token, and both are bad.

### Why not characters?

Characters are simple: about 100 symbols cover English text, and nothing is ever unknown. The problem is
length. English averages about 4 characters per token with a modern tokenizer, so a character-level model
reads sequences 4 times longer.

Longer sequences cost more in three ways:

- **Attention.** The core operation of a transformer (the network design all current LLMs use) compares every position with every earlier position
  ([Chapter 4](/foundations/transformer)). That work grows with the square of the length: 4 times longer
  means about 16 times more attention work.
- **Context.** A model reads at most a fixed number of positions at once. With characters, that window
  holds 4 times less text.
- **Generation.** A model writes one symbol per step. Characters mean 4 times as many steps for the same
  answer.

The model also has to spend its early layers assembling letters into words before it can do anything useful.

### Why not words?

Words are short sequences with clear meanings. The problem is that the set of words is open-ended. Names,
typos, URLs, code, new slang, numbers and other languages produce an endless supply of new words.

A word-level vocabulary must stop somewhere, say at the 50,000 most common words. Every other word becomes a
single "unknown" token, and its content is lost. This is the <Term id="oov">out-of-vocabulary (OOV)</Term> problem. A vocabulary
large enough to make it rare would need millions of entries, and each entry costs parameters, as you will see
below.

Words also hide useful structure. "run", "runs", "running" and "runner" would be four unrelated ids, so the
model learns each from scratch.

### The compromise: subwords

The answer used by every major LLM sits in between. Common words become one token. Rare words split into
a few common pieces: "unhappiness" might become "un" + "happiness". Anything else falls back to single bytes,
so no input is ever unknown. The next sections build such a tokenizer from the bottom up.

## Bytes and UTF-8

**In short:** every text is already a sequence of bytes, numbers from 0 to 255. Starting from bytes means
the tokenizer can encode any input at all.

Computers store text as bytes, using a standard encoding called <Term id="utf-8">UTF-8</Term>. Plain English letters, digits and
punctuation (the ASCII set) take 1 byte each. Other characters take 2 to 4 bytes:

```text
"a"   -> 1 byte    97
"é"   -> 2 bytes   195 169
"日"  -> 3 bytes   230 151 165
"🙂"  -> 4 bytes   240 159 153 130
```

So a vocabulary that contains the 256 possible byte values can spell any text in any language, plus emoji
and binary junk. Bytes are the safety net. The rest of the vocabulary consists of longer pieces that save
space on common text.

A side effect: a token boundary can fall in the middle of a character. A model that outputs only the first
byte of "日" has produced an incomplete character. Software that streams tokens to a screen must wait for the
rest of the bytes before it shows the character.

## Byte-pair encoding, step by step

**In short:** start with single bytes as tokens. Find the most frequent pair of adjacent tokens in a
training text, make it a new token, and repeat. The ordered list of merges is the tokenizer.

### A worked example

Take a tiny training text in which four words appear with these counts: "low" 5 times, "lower" 2,
"newest" 6 and "widest" 3. Start with each word split into single letters (for plain English, one letter is
one byte).

Now count every pair of adjacent symbols, weighting by how often its word appears. For example, "e s" appears
once in "newest" (6 times) and once in "widest" (3 times): 9 in total.

```text
pair    count   where
e s       9     newest (6), widest (3)
s t       9     newest (6), widest (3)
w e       8     lower (2), newest (6)
l o       7     low (5), lower (2)
o w       7     low (5), lower (2)
...
```

The most frequent pair wins. "e s" and "s t" tie at 9; take the first. Replace every "e s" with a new symbol
"es", then count again. Each round adds one token to the vocabulary:

```text
merge 1:  e + s    -> es      n e w es t,  w i d es t
merge 2:  es + t   -> est     n e w est,   w i d est
merge 3:  l + o    -> lo      lo w,        lo w e r
merge 4:  lo + w   -> low     low,         low e r
merge 5:  n + e    -> ne      ne w est
merge 6:  ne + w   -> new     new est
```

After 6 merges, "low" is one token, "newest" is "new" + "est", and "widest" is "w i d est". The suffix "est"
became a token because two different words share it. This is how BPE discovers useful word parts without
being told any grammar.

This procedure is <Term id="bpe">byte-pair encoding (BPE)</Term>. It began as a data compression trick
(Philip Gage, 1994) and was adapted
for translation models by [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909). You pick the number of
merges, and so the vocabulary size, in advance.

### Encoding new text

The trained tokenizer is two things: the 256 byte tokens and the ordered list of merges. To encode new text,
convert it to bytes and apply the merges in the order they were learned. Take "lowest": l o w e s t, then
merge 1 gives l o w es t, merge 2 gives l o w est, and merges 3 and 4 give low est. Two tokens, even though
"lowest" never appeared in the training text.

Decoding is easy: look up the bytes for each token id and join them. Encoding then decoding always gives back
the exact original text. Nothing is lost, which is a property word-level vocabularies never had.

::: details Going deeper: why the merge order matters
A token's bytes do not uniquely decide its pieces. "lowest" could also be cut as "lo" + "west" if both were
tokens. BPE resolves this by replaying the merges in training order, so the same text always gets the same
tokens. Some tokenizers instead pick, among all valid cuts, the one with the fewest tokens or the highest
probability; see the alternatives below.
:::

### Byte-level BPE and pre-tokenization

Real tokenizers add three details.

**Bytes, not characters.** Starting BPE from the 256 byte values, rather than from characters, is called <Term id="byte-level-bpe">byte-level BPE</Term>.
GPT-2 introduced it
([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)),
and most current models use it. Its vocabulary is 256 bytes + 50,000 merges + 1 special token = 50,257.

**A first split with a regular expression.** Without limits, BPE happily merges across word boundaries: in
English, "of the" is so common that " of the" could become a single token. Then the model has many
near-duplicate tokens such as "dog", "dog.", "dog!" and " dog,". So before BPE runs, a regular expression cuts
the text into chunks: a word with its leading space, a number, a run of punctuation, a run of whitespace.
Merges never cross chunk boundaries. This is <Term id="pre-tokenization">pre-tokenization</Term>.

Note that the space belongs to the following word. " cat" (with a leading space) and "cat" (at the start of a
line) are different tokens with different ids.

**Reserved tokens.** A <Term id="special-token">special token</Term> is a reserved id that no ordinary text can produce. GPT-2 has one,
`<|endoftext|>`, placed between documents during training. Chat models add more, to mark where a user turn or
an assistant turn starts ([Chapter 7](/building/post-training)). The tokenizer cuts them out of the input
before BPE runs, and must refuse to create them from user-typed text. Otherwise a user could type the string
for "assistant turn starts" and impersonate the model.

::: details Going deeper: the GPT-2 regex and the digit rules
GPT-2's pre-tokenization pattern is:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

In order: common English contractions, letters with an optional leading space, digits with an optional
leading space, punctuation runs, and whitespace. `\s+(?!\S)` keeps the last space of a run free, so it can
attach to the next word. GPT-4's tokenizer changed this to group digits in runs of at most 3 (`\p{N}{1,3}`),
and Llama 3 uses the same rule. Llama 1 and 2 split every number into single digits instead.

GPT-2 also maps each byte to a printable Unicode character before BPE, so the merge file is readable text. This
is why a space shows up as `Ġ` in GPT-2 vocabulary files.
:::

### How big is the vocabulary?

Vocabularies have grown. A larger vocabulary means fewer tokens per text, which is cheaper for everything that
scales with sequence length. It also covers more languages well. The cost is a bigger embedding table (below)
and more rare tokens that get little training.

| Model | Vocabulary size |
|---|---|
| GPT-2 (2019) | [50,257](https://huggingface.co/openai-community/gpt2/blob/main/config.json) |
| Llama 2 (2023) | 32,000 |
| Llama 3 (2024) | [128,256](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/blob/main/config.json) |
| DeepSeek-V3 (2024) | [129,280](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) |
| Mistral NeMo (2024) | [131,072](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/blob/main/config.json) |
| Qwen2.5 (2024) | [152,064](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json) |
| Gemma 3 (2025) | [262,208](https://huggingface.co/unsloth/gemma-3-4b-it/blob/main/config.json) |

These are the sizes of the models' embedding tables. Some include unused padding slots, rounded up so the
matrix shapes run fast on GPUs. OpenAI's GPT-4o tokenizer, `o200k_base`, has about 200,000 tokens
([tiktoken](https://github.com/openai/tiktoken)).

## Try it: train a BPE tokenizer

**In short:** about 130 lines of Python train a byte-level BPE on 1.1 MB of Shakespeare, then encode and
decode text with it.

The script downloads Tiny Shakespeare (about 1.1 MB, all of Shakespeare's plays in one file) on its first run,
and caches it in `exercises/data/`. It learns 512 merges, so the vocabulary is 256 + 512 + 1 special token =
769. It saves the merges to `exercises/data/bpe_merges.json`, and Chapter 4 trains a small transformer on
these tokens.

<<< @/../exercises/ch03_bpe.py

Output (about 25 seconds on 2 CPU cores):

```text
text: 1,115,394 characters, 1,115,394 bytes
first chunks: ['First', ' Citizen', ':', '\n', 'Before', ' we', ' proceed', ' any', ' further', ',', ' hear', ' me', ' speak', '.']

== Training 512 merges ==
merge   1: ' ' + 't' -> ' t'  (seen 23837 times)
merge   2: 'h' + 'e' -> 'he'  (seen 18203 times)
merge   3: ' ' + 'a' -> ' a'  (seen 13541 times)
merge   4: 'o' + 'u' -> 'ou'  (seen 12730 times)
merge   5: ' ' + 's' -> ' s'  (seen 12287 times)
merge   6: ' ' + 'm' -> ' m'  (seen 10786 times)
merge   7: 'i' + 'n' -> 'in'  (seen 10606 times)
merge   8: ' ' + 'w' -> ' w'  (seen 10546 times)
merge   9: 'r' + 'e' -> 're'  (seen 9843 times)
merge  10: 'h' + 'a' -> 'ha'  (seen 9673 times)
merge  11: 'n' + 'd' -> 'nd'  (seen 8730 times)
merge  12: ' t' + 'he' -> ' the'  (seen 8684 times)
merge 100: 'l' + 'y' -> 'ly'  (seen 1312 times)
merge 200: ' c' + 'om' -> ' com'  (seen 575 times)
merge 300: ' we' + 'll' -> ' well'  (seen 384 times)
merge 400: 'in' + 'ce' -> 'ince'  (seen 266 times)
merge 500: ' be' + 't' -> ' bet'  (seen 200 times)
trained in 33 s

vocabulary size: 769  (256 bytes + 512 merges + 1 special)
longest tokens: 'GLOUCESTER', ' brother', ' RICHARD', 'GLOUCEST', 'INCENTIO', ' should', ' father', ' friend'

== Encoding ==
whole text: 1,115,394 bytes -> 502,657 tokens  (2.22 bytes per token)

'To be, or not to be: that is the question.'
  42 bytes -> 15 tokens: ['To', ' be', ',', ' or', ' not', ' to', ' be', ':', ' that', ' is', ' the', ' qu', 'est', 'ion', '.']

"The king's horse ran 1234 miles."
  32 bytes -> 18 tokens: ['The', ' king', "'", 's', ' h', 'or', 'se', ' r', 'an', ' ', '1', '2', '3', '4', ' m', 'il', 'es', '.']

'Shakespeare, naïve café 日本'
  32 bytes -> 25 tokens: ['S', 'ha', 'k', 'es', 'p', 'ea', 're', ',', ' n', 'a', '�', '�', 've', ' c', 'a', 'f', '�', '�', ' ', '�', '�', '�', '�', '�', '�']

with a special token: [69, 120, 101, 117, 451, 46, 768, 65, 67, 84, 690]  -> 'Exeunt.<|endoftext|>ACT II'
round trip OK: decode(encode(s)) == s for every example
```

What to read from it:

- **The first merges are the most common pairs in English:** " t", "he", then " the" at merge 12. Later
  merges build whole words (" well") and common pieces ("ince").
- **The longest tokens come from the training text.** Character names in capitals, such as "GLOUCESTER",
  appear before every one of their lines, so they became tokens. A tokenizer reflects what it was trained on.
- **Compression is 2.2 bytes per token** with only 769 tokens. Production tokenizers with 100,000 or more
  tokens reach about 4 bytes per token on English.
- **Rare and foreign text falls apart.** "Shakespeare" barely appears in the plays, so it takes 7 tokens.
  "ï", "é" and "日本" never appear, so they fall back to raw bytes: 2 tokens each for the accented letters, 6
  for the two Japanese characters. Each `�` is one byte that is only part of a character. Decoding all the
  bytes together still restores the exact text.
- **"1234" is four tokens,** because this script's regex splits every digit, like Llama 2.

## Tokenizer quirks

**In short:** the model sees token ids, not letters or digits. Many famous LLM failures come from that, or
from a mismatch between what the tokenizer and the model were trained on.

### Counting letters

Ask a model how many r's are in "strawberry", and older models often said 2. The model never sees the letters.
It sees a few token ids, perhaps "str", "aw" and "berry", and each id is a number with no spelling attached.

To spell a word, the model must have learned the letters of each token indirectly, from text where words are
spelled out, and then count across token boundaries. It can learn this, but it is a much harder task than
it looks to a human reading the letters. Newer models do better, often by writing the word out letter by
letter first.

### Arithmetic and digits

GPT-2's regex keeps any run of digits together, and BPE then cuts it by frequency. "1234" might become
"12" + "34" in one place and "123" + "4" in another. The same digit then sits at different positions inside
different tokens, so the model cannot line up place values as you do when you add numbers column by column.
Common numbers such as years also became single tokens, while nearby rare numbers did not.

Fixes are in the pre-tokenization regex: split every digit (Llama 1 and 2), or group digits in fixed runs of
up to 3 (GPT-4, Llama 3). [Singh and Strouse (2024)](https://arxiv.org/abs/2402.14903) found that the
direction matters: grouping 3 digits from the right, as people write thousands separators, made GPT-3.5 and
GPT-4 noticeably better at addition than grouping from the left.

### Whitespace and code

GPT-2 had no tokens for runs of spaces, so each space of Python indentation cost a token. Code was expensive
and the model saw less of it per context window. Later tokenizers added tokens for runs of spaces and tabs.

A related trap: because the space belongs to the following word, a prompt ending in a space ("The capital of
France is ") leaves the model in an odd position. In training, the space almost always came glued to the next
word as " Paris", so a lone trailing space is rare, and outputs after it are worse.

### The non-English token tax

A tokenizer gets most of its merges from the language that dominates its training text, usually English.
Other languages, especially those in other scripts, get fewer merges and fall back to smaller pieces, down to
single bytes.

[Petrov et al. (2023)](https://arxiv.org/abs/2305.15425) translated the same sentences into many languages and
counted tokens. With the tokenizers they studied, some languages needed up to 15 times as many tokens as
English for the same content. Speakers of those languages pay more per request, wait longer for answers, and
fit less text into the context window. Larger multilingual vocabularies, such as Gemma's 262K, shrink this gap
but do not close it.

### Glitch tokens

In 2023, researchers found that GPT-2 and GPT-3 behaved strangely when asked to repeat certain tokens, such as
" SolidGoldMagikarp" ([Rumbelow and Watkins, 2023](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)).
The model would refuse, answer with a different word, or produce insults.

The cause is a mismatch. The tokenizer was trained on data where " SolidGoldMagikarp" was common: it was the
username of a very active poster in a Reddit counting forum. The model was trained on different, filtered data
in which that string almost never appeared. So the token exists, but the model has almost never seen it. As
the next section explains, each token's vector learns only from text that contains that token. This one
stayed close to its random starting point, and the model has no idea what it means.

Such a token is a <Term id="glitch-token">glitch token</Term>. Every model has some, and they can be found automatically by
looking for embedding vectors that barely moved during training
([Land and Bartolo, 2024](https://arxiv.org/abs/2405.05417)).

### Alternatives to BPE

- **Unigram and SentencePiece.** The <Term id="unigram-tokenizer">unigram tokenizer</Term> works the other way round: start
  with a large vocabulary and prune it, keeping the tokens that best explain the text under a simple
  probability model ([Kudo, 2018](https://arxiv.org/abs/1804.10959)). SentencePiece
  ([Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)) is a library that trains unigram or BPE
  tokenizers directly on raw text, treating the space as an ordinary symbol. Llama 1 and 2 and Gemma use
  SentencePiece.
- **No tokenizer at all (research).** Byte-level models read raw bytes and avoid every quirk above, at the
  cost of longer sequences. Recent designs group bytes into variable-size patches inside the model to win the
  length back: [MEGABYTE (2023)](https://arxiv.org/abs/2305.07185) and the Byte Latent Transformer
  ([Pagnoni et al., 2024](https://arxiv.org/abs/2412.09871)), which matched a BPE-based Llama 3 at
  8B-parameter scale in its authors' tests. As of 2026, frontier models still use subword tokenizers.

## Embeddings: a lookup table that learns

**In short:** each token id selects one row of a learned matrix. That row, a vector of a few thousand
numbers, is how the model sees the token. Training shapes the rows so that tokens used alike end up close.

### From an id to a vector

A token id is a label, not a quantity. Token 902 is not "bigger" than token 17, so feeding the raw id into a
multiply would be meaningless. The standard fix has two steps.

First, turn the id into a <Term id="one-hot">one-hot vector</Term>: a vector as long as the vocabulary, all zeros except a 1 at
position id. With a vocabulary of 5 and id 2, that is [0, 0, 1, 0, 0]. Every token is now equally different
from every other.

Second, multiply by a learned matrix E with one row per vocabulary entry and d columns:

```text
one_hot(id) × E  =  row number id of E          one_hot: 1 × V,  E: V × d,  result: 1 × d

[0 0 1 0 0] × | 0.1  0.7 |
              | 0.4 −0.2 |
              | 0.9  0.3 |   =  [0.9  0.3]
              |−0.5  0.0 |
              | 0.2  0.8 |
```

V is the vocabulary size and d is the model's width, the length of the vector that represents each position.
Multiplying by a one-hot vector picks out one row: every other row gets multiplied by 0. So in code nobody
does the multiply. They read the row, like an array lookup. The row is the token's <Term id="embedding">embedding</Term>, and E is the
embedding table.

<TokPipelineDiagram />

### Learned by backpropagation

E starts as small random numbers, like any other weight matrix ([Chapter 2](/foundations/learning)). Seen as
a multiply by a one-hot vector, the lookup is an ordinary layer, so backpropagation computes a gradient for E
and the optimizer updates it.

There is one special property. In a batch, only the rows of tokens that appear in that batch get a non-zero
gradient. Other rows have a 0 in the one-hot vector, so the loss does not depend on them. A token that never
appears in training is never updated. That is the glitch-token story from above, seen from the inside.

### Geometry: similar tokens end up close

Why would the rows mean anything? Think about what the gradient asks of each row. The model is trained to
predict the next token. If "cat" and "dog" appear in similar contexts ("the ___ sat", "feed the ___"), then
their rows receive similar gradient pushes, and they drift toward similar values. Tokens used alike end up
close, measured by <Term id="cosine-similarity">cosine similarity</Term>, the angle between two vectors.

The most famous demonstration comes from word2vec ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)), a
small model trained only to learn word vectors. Its vectors showed offsets that look like relationships:

```text
vector("king") − vector("man") + vector("woman")  ≈  vector("queen")
```

Treat this as a simplification. The famous results exclude the three input words from the answer. Without
that rule, the nearest vector is often "king" itself
([Nissim et al., 2019](https://arxiv.org/abs/1905.09866)). Clean offsets hold for some relationships, such
as capitals and countries, and fail for many others. And in an LLM, the input embedding is only a starting
point. A word's meaning in context ("bank" of a river, or a bank holding money) is built up by the layers above
([Chapter 4](/foundations/transformer)).

### The way out: unembedding and weight tying

At the other end, the model must turn its final vector for a position (length d) back into a score for each
of the V tokens. It does this with another V × d matrix U, the <Term id="unembedding">unembedding</Term>:

```text
logit_t = h · U_t        h: final vector (length d),  U_t: row t of U,  t = 0 … V−1
```

The score for token t is the dot product of h with token t's row: how well h points in that token's
direction. These scores are the logits.

E and U have the same shape. <Term id="weight-tying">Weight tying</Term> uses one matrix for both
([Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)). It saves V × d parameters, and it makes sense:
a token's input vector and the direction that predicts it are related. GPT-2 and small models such as Llama
3.2 1B tie their weights. Larger models, such as Llama 3 8B, Qwen2.5 7B and DeepSeek-V3, keep them separate
(see their configs, linked in the table above). At their size the saving is a small fraction, and separate
matrices give more freedom.

::: info Back-of-envelope
**Embedding parameters for Llama 3 8B.** Vocabulary 128,256, width 4,096:

```text
128,256 × 4,096 ≈ 525 million parameters per matrix
untied: embedding + unembedding ≈ 1.05 billion ≈ 13% of the 8B total
memory in bf16 (2 bytes each): ≈ 1.05 GB per matrix
```

The unembedding is also compute, not just memory: it is a full matrix multiply for every generated token.
About 2 × 4,096 × 128,256 ≈ 1 GFLOP (a billion multiply-or-add operations) per token, against roughly 2 × 8B = 16 GFLOP for the whole forward pass:
about 6%. The embedding lookup is nearly free: reading one row is no multiply at all.

**Tokens per word.** A common rule of thumb for English with GPT-style tokenizers: 1 token ≈ 4 characters ≈
0.75 words, so 1,000 words ≈ 1,300 tokens. The same text in a language the tokenizer serves poorly can cost
several times as many tokens, up to about 15 times in Petrov et al.'s worst cases. A 128K-token context window
holds about 100,000 English words, and far fewer in those languages.
:::

## From vectors to the next token

**In short:** a language model outputs a probability for every token in the vocabulary, as a guess for the
next one. Its loss is the average surprise at the real next token, and perplexity is that loss, made readable.

Put the pieces together. Text becomes token ids. Ids become embedding vectors. The model (Chapter 4) turns the
vectors into a final vector per position. The unembedding turns that into V logits, and <Term id="softmax">softmax</Term> turns
the logits into a probability for each possible next token.

That is the whole interface of a <Term id="language-model">language model</Term>: given the tokens so far, a probability
distribution over the next token. Training uses the <Term id="cross-entropy">cross-entropy</Term> loss from Chapter 2:
−log of the probability given to the token that actually came next, averaged over every position of the
training text. To generate text, sample one token from the distribution, append it, and repeat.

### Perplexity

A loss of 2.5 is hard to picture. <Term id="perplexity">Perplexity</Term> converts it:

```text
perplexity = exp(average loss per token)
```

It reads as "the model is as unsure as if it were choosing evenly among this many tokens". A model that
guesses uniformly over V tokens has loss ln(V) and perplexity exactly V. A perfect model has perplexity 1.

Perplexity has a catch: it is per token, and tokens differ between tokenizers. A model with long tokens makes
fewer, harder predictions, so its per-token perplexity is higher even if it predicts the text better. To
compare models with different tokenizers, divide the total loss by the number of bytes (or characters)
instead. The exercise below shows this.

## Try it: a bigram language model

**In short:** the smallest possible language model predicts the next token from the current one only. Its
whole network is one embedding table whose rows are next-token logits.

A <Term id="bigram-model">bigram model</Term> looks at one token and guesses the next. Make the embedding table V × V instead
of V × d: then the row for the current token has one number per vocabulary entry, and those numbers can be the
logits directly. No other layers.

The script trains this model by gradient descent, first on raw bytes (V = 256), then on the 769 BPE tokens
from the previous exercise. Each step takes 4,096 random positions from the first 90% of the text; the last
10% is held out as validation data. Then it samples text and checks which rows ended up close.

<<< @/../exercises/ch03_bigram.py

Output (about 70 seconds on 2 CPU cores):

```text
== Bytes: vocab 256, 1,115,394 tokens, 65,536 parameters ==
guessing uniformly: loss = ln(256) = 5.55
step    0  train loss 6.090  val loss 5.913
step  300  train loss 2.466  val loss 2.498
step  600  train loss 2.447  val loss 2.500
step  900  train loss 2.455  val loss 2.499
step 1200  train loss 2.453  val loss 2.496
step 1500  train loss 2.484  val loss 2.499
val perplexity 12.2   val loss per byte 2.499 nats
sample:

Ar, isehag cona,
I menodes O:

So hees my arsun woup heer ber,
S:
Thely four.
n we?
ARBRKINCofundoloveparite grd n t.
Whe avet yollikigoy mard?
NG con

== BPE tokens: vocab 769, 502,657 tokens, 591,361 parameters ==
guessing uniformly: loss = ln(769) = 6.65
step    0  train loss 7.172  val loss 7.104
step  300  train loss 3.576  val loss 4.051
step  600  train loss 3.586  val loss 4.085
step  900  train loss 3.606  val loss 4.106
step 1200  train loss 3.584  val loss 4.132
step 1500  train loss 3.582  val loss 4.145
val perplexity 63.1   val loss per byte 1.942 nats
sample:
First Murdeed me, as by my ban, other decest thou may be king.
Sirliness.
BROMARET:

That Aull make hasty-hedom I may comms the in that?



== Nearest rows (cosine similarity) ==
' he'     -> ' we', ' they', ' I', ' it', 'He'
' my'     -> ' our', ' your', ' thy', ' his', ' the'
' the'    -> ' my', ' your', ' his', 'The', ' our'
' king'   -> ' day', ' word', 'man', 'urse', 'io'
' love'   -> 'urse', 'ase', 'ire', 'ake', ' fear'
```

What to read from it:

- **The loss starts above ln(V)**, the loss of uniform guessing, because the random starting logits are
  slightly confident in random directions. It falls fast: a bigram model has little to learn.
- **The byte model plateaus at 2.50 on both train and validation.** It has learned the byte-pair statistics
  of English and nothing more. Its samples have word-like shapes but few real words.
- **The BPE model overfits.** Its validation loss is lowest at the first checkpoint (step 300), then rises while training loss
  stays flat. It has 591,361 parameters for about 450,000 training tokens, and the rows of rare tokens memorise
  their few training examples.
- **Compare per byte, not per token.** The BPE perplexity (63) looks far worse than the byte one (12), but
  per byte the BPE model is better: 1.94 nats against 2.50 (a nat is the unit of a loss computed with the
  natural log). Each BPE token covers about 2.2 bytes, so
  "previous token" is more context than "previous byte". Its samples contain real words.
- **Rows that behave alike end up close.** " he" sits near " we", " they", " I"; " my" near " our",
  " your", " thy", " his". Nobody told the model about pronouns: these tokens are followed by similar tokens,
  so their rows got similar gradients. Rarer tokens such as " king" and " love" have noisier neighbours.

Chapter 4 keeps the tokenizer and the training loop and replaces the single table with a transformer, which
looks at all previous tokens instead of one.

::: warning Thread
The tokenizer is the most frozen part of a model. It is fixed before pretraining starts, and every embedding
row is tied to it. A word that becomes important after training, such as a new product name, is always split
into old pieces, and the model must rebuild its meaning from context every time it reads it. Glitch tokens
are an example of the "tricked the same way forever" intuition: the same input breaks the same model until
someone retrains it.

The counterpoints are strong here. Byte-level BPE can represent any text, so nothing new is unreadable, and
models handle new words well in context. Vocabularies can be extended after training by adding rows and
training further on new text, a common way to adapt a model to a new language. The limit is not representation but how much
the model knows about the new pieces, which brings the question back to learning in the weights.
:::

::: tip Interview checkpoint
<details><summary>Walk me through how a BPE tokenizer is trained and used.</summary>

Split the training text into chunks with a regex (words, numbers, punctuation, whitespace), then into bytes.
Repeatedly count adjacent pairs, merge the most frequent into a new token, and record the merge, until the
vocabulary reaches the target size. To encode, convert text to bytes and replay the merges in order. To
decode, concatenate each token's bytes. Starting from bytes means any input can be encoded.
</details>

<details><summary>What are the trade-offs in choosing vocabulary size?</summary>

Bigger vocabularies give fewer tokens per text: cheaper attention, more text per context window, fewer
generation steps, and better coverage of other languages. The costs are V × d parameters in the embedding and
again in the unembedding, a larger softmax per generated token, and more rare tokens whose rows get little
training. Current models sit between about 32K and 260K.
</details>

<details><summary>Why are LLMs bad at counting letters and at long arithmetic?</summary>

They see token ids, not characters, so the letters inside a token are not visible and must have been learned
indirectly. For numbers, older tokenizers split digit runs inconsistently, so place values do not line up.
Splitting digits individually or in fixed groups of 3 helps.
</details>

<details><summary>Model A has perplexity 8 and model B has perplexity 12 on the same text. Is A better?</summary>

Only if they share a tokenizer. Perplexity is per token, and a tokenizer with longer tokens makes fewer but
harder predictions. Compare total loss divided by bytes or characters (bits per byte) instead.
</details>

<details><summary>What is weight tying, and why do large models often skip it?</summary>

Using the same V × d matrix for the input embedding and the output unembedding. It saves V × d parameters,
which matters for small models where the embedding can be a large share of all weights. For large models the
saving is a few percent, and separate matrices give the model more freedom.
</details>
:::

## In practice

- **Everything is billed and limited in tokens.** API prices, rate limits and context windows count tokens,
  so the same text costs different amounts on different models. Count with the model's own tokenizer
  (`tiktoken` for OpenAI models, the Hugging Face tokenizer for open models).
- **Tokenizer and model ship together.** Using a model with the wrong tokenizer gives ids that mean other
  tokens, and fluent-looking garbage. Many silent bugs in fine-tuning pipelines come from a mismatched
  tokenizer or chat template (the format that wraps each message in special tokens).
- **Special tokens are a security boundary.** Chat formats mark roles with special tokens. Tokenizers must
  never produce them from user text, or a user can forge an assistant or system turn.
- **Tokenizer training data matters.** Teams train the tokenizer on a sample that matches the planned
  training mixture, and check for tokens that will rarely appear in it, to avoid glitch tokens.

## Common misconceptions

- **"A token is a word."** It is often a word with its leading space, often part of a word, and sometimes a
  single byte. About 0.75 English words per token on average.
- **"The model reads characters."** It reads ids. Spelling is something it has to learn about each token.
- **"Embeddings are a separate model."** In an LLM, the embedding table is the first layer of the same
  network, trained with the rest by the same loss.
- **"Lower perplexity always means a better model."** Only with the same tokenizer and the same test text.
- **"A bigger vocabulary is always better."** It shortens sequences but costs parameters and leaves more
  under-trained rare tokens.

## Key takeaways

- Text becomes token ids through a fixed tokenizer. Characters make sequences too long; words cannot cover an
  open vocabulary. Subwords sit in between.
- BPE starts from the 256 byte values and repeatedly merges the most frequent adjacent pair. Byte-level BPE
  can encode anything, and decoding restores the exact text.
- Many LLM quirks are tokenizer effects: weak letter counting, digit-splitting arithmetic problems, higher
  token counts for non-English text, and glitch tokens whose rows were never trained.
- An embedding is a row of a learned V × d table: a one-hot multiply, done as a lookup. Tokens used alike end
  up with nearby rows. The unembedding maps the final vector back to V logits.
- A language model outputs a distribution over the next token. Its loss is cross-entropy; perplexity is
  exp(loss), comparable only under the same tokenizer.

## Further reading

- [Andrej Karpathy, *Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) and
  [minbpe](https://github.com/karpathy/minbpe): builds a GPT-4-style tokenizer live; the exercise follows a
  simpler version of the same path.
- [Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909):
  the paper that brought BPE to neural networks; short and clear.
- [Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
  section 2.2 explains byte-level BPE.
- [Rumbelow and Watkins, *SolidGoldMagikarp*](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation):
  the discovery of glitch tokens, with many examples.
- [Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (2023)](https://arxiv.org/abs/2305.15425):
  measured token costs across languages.
- [Mikolov et al., *word2vec* (2013)](https://arxiv.org/abs/1301.3781) and
  [Nissim et al., *Fair is Better than Sensational* (2019)](https://arxiv.org/abs/1905.09866): the famous
  analogies, and why to take them with care.

## Review

<Flashcards id="tokens-embeddings" :cards="cards" />

<MarkDone id="tokens-embeddings" />

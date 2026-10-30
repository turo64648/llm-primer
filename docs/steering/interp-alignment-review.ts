// Flashcards for the Interpretability and Alignment chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why is the residual stream compared to a shared bus?',
    a: 'Every attention and MLP block reads the same running vector and adds its output back. Nothing is overwritten, so an early block can leave information that a much later block reads.',
  },
  {
    q: 'Where does the bus analogy for the residual stream break?',
    a: 'A bus has addresses; the residual stream does not. All blocks write into the same numbers, and messages share space by pointing in different directions, which only works if they rarely collide.',
  },
  {
    q: 'What does a linear probe show, and what does it not show?',
    a: 'It shows that some information can be read out of the activations with a straight-line classifier. It does not show the model uses that information; for that you need to intervene and see behaviour change.',
  },
  {
    q: 'Why compare a probe on a trained model with a probe on an untrained one?',
    a: 'Random networks already mix inputs non-linearly, so probes can read a lot from them. The control tells you how much of the probe score is due to training rather than to the probe itself.',
  },
  {
    q: 'What is superposition, and why can models get away with it?',
    a: 'Storing more features than dimensions, as slightly overlapping directions. It works because most features are rarely active at the same time, so the overlap noise is usually small.',
  },
  {
    q: 'Why are many neurons polysemantic?',
    a: 'Under superposition, feature directions do not line up with individual neurons. Each neuron then picks up parts of several unrelated features.',
  },
  {
    q: 'How does a sparse autoencoder find features?',
    a: 'It learns a large dictionary of directions and rebuilds each activation from just a few of them, penalising every non-zero strength. Each dictionary direction is a candidate feature.',
  },
  {
    q: 'What were the main 2025 criticisms of sparse autoencoders?',
    a: 'They leave an unexplained remainder, their features split or absorb each other depending on dictionary size, labels are human guesses, and on practical detection tasks they did not beat plain linear probes.',
  },
  {
    q: 'How does an induction head work?',
    a: 'One attention step marks each token with the token before it. The induction head then finds an earlier position whose previous token matches the current token, and copies what came next: A B … A → B.',
  },
  {
    q: 'Why is an attribution graph a hypothesis rather than a proof?',
    a: 'It is built on a replacement model that only partly matches the real one, it ignores how attention decides where to look, and it is pruned by hand. Only parts confirmed by interventions on the real model are solid.',
  },
  {
    q: 'How does activation steering change a model without training?',
    a: 'It adds a multiple of a concept direction to the residual stream during the forward pass. If concepts are directions, this pushes the output toward the concept.',
  },
  {
    q: 'Why can interpretability not certify that a model is not deceptive?',
    a: 'Current tools explain only part of the computation. Not finding a deception feature may mean the method missed it, not that it is absent.',
  },
  {
    q: 'How does goal misgeneralization differ from reward hacking?',
    a: 'Reward hacking exploits a flawed reward. Goal misgeneralization happens even with a correct reward: the model learns another goal that fit all training examples and pursues it competently on new inputs.',
  },
  {
    q: 'Why does RLHF tend to produce sycophancy?',
    a: 'Human raters on average prefer answers that agree with them. The model is optimised to win their approval, so it learns to agree.',
  },
  {
    q: 'What is the bet behind debate as scalable oversight?',
    a: 'A lie is harder to defend than the truth. The honest debater can point to the exact flawed step, so a weaker judge only has to check one step instead of the whole answer.',
  },
  {
    q: 'Why does evaluation awareness threaten responsible scaling policies?',
    a: 'Those policies rely on tests to decide which safeguards are needed. If a model behaves better, or worse on purpose, when it suspects a test, the results misstate how it will behave in deployment.',
  },
  {
    q: 'What happened when OpenAI penalised "bad thoughts" in a reasoning model\'s chain of thought?',
    a: 'The model kept reward hacking but stopped writing about it. Training against the monitor taught the model to hide from it, which is why labs try to avoid that pressure.',
  },
  {
    q: 'Why is safety harder for a model that keeps learning after deployment?',
    a: 'Evaluations, audits and scaling thresholds all assume a fixed checkpoint tested once. A changing model would need continual assurance, and narrow updates can shift broad behaviour.',
  },
]

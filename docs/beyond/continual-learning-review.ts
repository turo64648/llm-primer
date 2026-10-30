// Flashcards for the Continual Learning chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why do labs ship frozen checkpoints instead of models that keep learning?',
    a: 'A fixed set of weights can be evaluated and signed off once, reproduced exactly, cached and served to millions of users as one copy, and rolled back. A model that changes after every conversation loses all of that and invites poisoning.',
  },
  {
    q: 'How do deployed LLMs get new knowledge today?',
    a: 'Through new releases: continued pretraining on newer text, fresh rounds of post-training (often using feedback collected from users), and customer fine-tunes. Between releases, new facts come in only through the context: prompts, retrieval and tools.',
  },
  {
    q: 'What causes catastrophic forgetting?',
    a: 'All tasks share the same weights. Gradient descent on the new task moves weights in whatever direction lowers the new loss, with no term that protects the old one, so knowledge stored in those weights is overwritten.',
  },
  {
    q: 'How do gradients tell you whether learning task B will hurt task A?',
    a: 'A small step against B’s gradient changes A’s loss by about −η times the dot product of the two gradients. If they point in opposing directions (negative dot product), each step for B raises A’s loss.',
  },
  {
    q: 'What is the stability–plasticity dilemma?',
    a: 'A learner must change to absorb new things (plasticity) but stay put to keep old things (stability). Every continual-learning method picks a point on that trade-off; none removes it.',
  },
  {
    q: 'How does EWC decide which weights to protect?',
    a: 'It measures each weight’s Fisher information on the old task: how sharply the model’s predictions change when that weight moves. It then penalises moving important weights away from their old values, more for more important ones.',
  },
  {
    q: 'Why does replay work so well, and why is it awkward at LLM scale?',
    a: 'It keeps real old-task gradients in every batch, so the optimiser balances both tasks as in joint training. At scale you must store or regenerate representative old data and pay extra compute for every update.',
  },
  {
    q: 'How do parameter-isolation methods avoid forgetting, and what do they give up?',
    a: 'Each task gets its own parameters, such as a LoRA adapter, so training one cannot overwrite another. They give up sharing: what is learned in one adapter does not improve the others, and you must pick the right adapter at run time.',
  },
  {
    q: 'What did “LoRA learns less and forgets less” find?',
    a: 'On code and maths, LoRA learned the new domain less well than full fine-tuning but kept more of the base model’s other abilities. Restricting how far weights can move trades plasticity for stability.',
  },
  {
    q: 'How is loss of plasticity different from catastrophic forgetting?',
    a: 'Forgetting loses old skills. Loss of plasticity loses the ability to learn new ones: after many tasks, a network trained by plain gradient descent learns each new task worse, as units die and weights grow large.',
  },
  {
    q: 'What does complementary learning systems theory say?',
    a: 'The brain pairs a fast learner, the hippocampus, which stores episodes after one exposure without disturbing old memories, with a slow learner, the neocortex, which extracts structure from many replayed, interleaved episodes, partly during sleep.',
  },
  {
    q: 'Why would a brain need a slow learner at all?',
    a: 'A network that learns new things fast in shared weights overwrites old knowledge. Learning slowly from interleaved old and new examples avoids that and finds structure shared across experiences. The fast store holds new things until then.',
  },
  {
    q: 'What is missing between an LLM’s context and its weights?',
    a: 'Anything that learns on a scale of hours to months. Activations last one token and context lasts one conversation; weights change only at the next release. Brains have synaptic changes at every timescale in between.',
  },
  {
    q: 'Why is a model per user hard, even with cheap LoRA adapters?',
    a: 'Storage and serving are manageable (adapters are tens of MB and many can share one base copy). The hard parts are the training compute per update, privacy of what each adapter memorises, and evaluating millions of different models for safety.',
  },
  {
    q: 'Why does continuous learning make safety harder?',
    a: 'Today a model is evaluated once and then does not change. A model that updates keeps drifting away from what was tested, can be poisoned by crafted inputs, and needs continuous evaluation and a way to roll back.',
  },
  {
    q: 'What is the strongest case that frozen models are “continual enough”?',
    a: 'Releases every few months, plus in-context learning, long context and retrieval, cover much of what users need; feedback from deployment already flows into the next release. The open question is whether thinking better, not only knowing more, needs faster weight updates.',
  },
]

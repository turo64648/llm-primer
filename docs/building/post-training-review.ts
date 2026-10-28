// Flashcards for the Post-training and Reasoning chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'Why is a pretrained base model not an assistant?',
    a: 'It was trained to imitate documents, so it continues text in whatever style fits, good or bad, and may answer with more questions. It has no chat format and no preference for helpful replies.',
  },
  {
    q: 'What does SFT change compared with pretraining?',
    a: 'Only the data: curated conversations in a fixed chat template instead of web text. The loss is the same next-token loss, counted only on the assistant’s tokens.',
  },
  {
    q: 'Why mask the loss on prompt tokens during SFT?',
    a: 'You want the model to learn to write replies, not to write user prompts. The prompt still goes through the model as context; it just produces no training signal.',
  },
  {
    q: 'Why does SFT data quality matter more than quantity?',
    a: 'The base model already has most of the knowledge; SFT mainly teaches format and behaviour. Bad examples teach bad habits, and LIMA got strong results from 1,000 good ones.',
  },
  {
    q: 'How does the policy gradient improve a model without being shown the right answer?',
    a: 'It samples a reply, scores it, and makes that reply more likely if it scored above expectation and less likely if below. Over many samples, good behaviour becomes more probable.',
  },
  {
    q: 'Why use an advantage (reward minus baseline) instead of the raw reward?',
    a: 'If all rewards are high, raw rewards push every reply up and the signal is noise. Subtracting the expected score means only better-than-usual replies are reinforced.',
  },
  {
    q: 'How do you train a reward model when humans only say “A is better than B”?',
    a: 'Use Bradley–Terry: model P(A beats B) as sigmoid(r(A) − r(B)) and maximise the probability of the choices humans made. Only score differences matter.',
  },
  {
    q: 'Why does RLHF penalise KL divergence from the reference model?',
    a: 'The reward model is only accurate near the outputs it was trained on. The penalty keeps the policy there, which limits reward hacking and keeps the language fluent.',
  },
  {
    q: 'What is reward hacking? Give two examples.',
    a: 'Raising the reward by exploiting flaws in the judge instead of improving. Examples: RLHF models becoming verbose or sycophantic, and coding models editing unit tests.',
  },
  {
    q: 'Why does PPO-based RLHF need four models in memory?',
    a: 'The trained policy, a trained value model for the baseline, a frozen reference model for the KL penalty, and a frozen reward model as the judge. For 70B that is about 2.5 TB of state.',
  },
  {
    q: 'How does DPO get rid of the reward model?',
    a: 'The best policy for the KL-regularised RLHF goal has a closed form, so the reward equals β·log(π/π_ref) plus a prompt-only term. In the Bradley–Terry loss that term cancels, leaving a loss on the policy alone.',
  },
  {
    q: 'What does DPO give up compared with online RL?',
    a: 'It trains on a fixed set of pairs and never samples from the current model, so it cannot learn from its own new mistakes. Iterative DPO partly fixes this.',
  },
  {
    q: 'How does GRPO compute advantages without a value model?',
    a: 'It samples a group of replies to the same prompt and scores each one relative to the group: (reward − group mean) / group std.',
  },
  {
    q: 'Why does GRPO learn nothing from very easy or very hard prompts?',
    a: 'If every reply in the group gets the same reward, all advantages are zero. Training needs problems the model sometimes solves.',
  },
  {
    q: 'Why do verifiable rewards allow longer, harder RL than RLHF?',
    a: 'A program that checks answers is much harder to fool than a learned reward model, so the policy can move further from its start without the reward becoming meaningless.',
  },
  {
    q: 'Why does a chain of thought improve answers?',
    a: 'Each written token is another pass through the network, and the model can read its own earlier steps. So it gets more computation and scratch memory per problem.',
  },
  {
    q: 'Does RL teach reasoning models new abilities?',
    a: 'Contested. At large pass@k, base models often match RL models, suggesting RL mostly sharpens existing paths; ProRL reports real gains after long RL training.',
  },
  {
    q: 'Why does LoRA save so much memory?',
    a: 'It freezes the original weights and trains only a low-rank delta B·A per matrix, often under 1% of the parameters. Gradients and optimizer state exist only for that small delta.',
  },
]

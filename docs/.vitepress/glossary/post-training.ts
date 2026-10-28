import type { GlossaryEntry } from '../glossary'

const chapter = '/building/post-training'

export const terms: Record<string, GlossaryEntry> = {
  'base-model': { term: 'Base model', def: 'The raw model straight out of pretraining. It continues any text it is given but does not reliably follow instructions or act as an assistant.', chapter },
  'post-training': { term: 'Post-training', def: 'All training done after pretraining to make a model useful: supervised fine-tuning, preference tuning (RLHF, DPO) and reinforcement learning.', chapter },
  sft: { term: 'Supervised fine-tuning (SFT)', def: 'Further training on example conversations with ideal replies, using the same next-token loss as pretraining. Teaches format and style.', chapter },
  'chat-template': { term: 'Chat template', def: 'The fixed format, with special marker tokens, that turns a conversation of system, user and assistant turns into one token sequence.', chapter },
  'loss-masking': { term: 'Loss masking', def: 'Counting the training loss only on some tokens, such as the assistant’s reply, so the model reads the prompt but is not trained to write it.', chapter },
  'reinforcement-learning': { term: 'Reinforcement learning (RL)', def: 'Training by trial and score: the model tries something, gets a reward, and is updated to make high-reward behaviour more likely.', chapter },
  policy: { term: 'Policy', def: 'In reinforcement learning, the thing that chooses actions. For an LLM, the model itself, seen as a probability distribution over replies.', chapter },
  reward: { term: 'Reward', def: 'A single number scoring how good an action or reply was. It can come from a human, a reward model, or a program that checks the answer.', chapter },
  'policy-gradient': { term: 'Policy gradient', def: 'An RL update that makes each sampled reply more or less likely in proportion to its advantage. REINFORCE, PPO and GRPO are all policy-gradient methods.', chapter },
  advantage: { term: 'Advantage', def: 'How much better a reply scored than expected: reward minus a baseline. Positive advantage makes the reply more likely; negative makes it less likely.', chapter },
  'reward-model': { term: 'Reward model', def: 'A network trained on human or AI preference pairs to output a score for a reply. Used as the judge during RLHF.', chapter },
  'bradley-terry': { term: 'Bradley–Terry model', def: 'A model of pairwise comparisons: the chance A beats B is sigmoid(score A − score B). Used to train reward models from preferences.', chapter },
  rlhf: { term: 'RLHF', def: 'Reinforcement learning from human feedback: train a reward model on human preference pairs, then optimise the model against it with RL and a KL penalty.', chapter },
  ppo: { term: 'PPO', def: 'Proximal policy optimization: a policy-gradient algorithm with a learned value model as baseline and clipped updates to keep each step small.', chapter },
  'kl-divergence': { term: 'KL divergence', def: 'A measure of how different two probability distributions are; zero when identical. In RLHF it penalises drifting away from the reference model.', chapter },
  'value-model': { term: 'Value model', def: 'In PPO, a second network trained to predict the expected reward, used as the baseline for advantages. GRPO removes it.', chapter },
  'reward-hacking': { term: 'Reward hacking', def: 'When a model raises its reward by exploiting flaws in the reward signal, such as verbosity or gaming tests, without getting better at the real task.', chapter },
  dpo: { term: 'DPO', def: 'Direct preference optimization: trains directly on preference pairs with a classification-style loss, with no reward model and no sampling during training.', chapter },
  rlaif: { term: 'RLAIF', def: 'Reinforcement learning from AI feedback: like RLHF, but a model rather than a human judges which reply is better.', chapter },
  'constitutional-ai': { term: 'Constitutional AI', def: 'Anthropic’s method where a model critiques and revises replies, and an AI judge ranks them, against a written list of principles.', chapter },
  rlvr: { term: 'RLVR', def: 'Reinforcement learning with verifiable rewards: RL where a program checks the answer, as in math problems or code with tests.', chapter },
  grpo: { term: 'GRPO', def: 'Group relative policy optimization: samples a group of replies per prompt and uses each reply’s score relative to the group as its advantage, with no value model.', chapter },
  'chain-of-thought': { term: 'Chain of thought', def: 'Intermediate reasoning steps a model writes before its final answer. The written tokens act as extra computation and scratch memory.', chapter },
  'reasoning-model': { term: 'Reasoning model', def: 'A model trained, usually with RL on verifiable rewards, to think at length before answering, such as OpenAI o1 or DeepSeek-R1.', chapter },
  'test-time-compute': { term: 'Test-time compute', def: 'Computation spent while answering rather than during training: longer reasoning, or sampling many answers and picking the best.', chapter },
  distillation: { term: 'Distillation', def: 'Training a smaller model on outputs generated by a larger, stronger model, so the small model picks up its behaviour cheaply.', chapter },
  lora: { term: 'LoRA', def: 'Low-rank adaptation: freezes the original weights and trains a small correction B·A for each matrix, cutting fine-tuning memory sharply.', chapter },
  peft: { term: 'PEFT', def: 'Parameter-efficient fine-tuning: methods such as LoRA that adapt a model by training only a small fraction of its parameters.', chapter },
}

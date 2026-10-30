import type { GlossaryEntry } from '../glossary'

const chapter = '/beyond/beyond-frozen-weights'

export const terms: Record<string, GlossaryEntry> = {
  'fast-weights': { term: 'Fast weights', def: 'Weights that change quickly while the network runs, holding recent information, on top of the slow weights learned in training. Linear attention and test-time memories are forms of fast weights.', chapter },
  'associative-memory': { term: 'Associative memory', def: 'A memory that stores key→value pairs and, given a key (or something close to it), returns the matching value.', chapter },
  'outer-product': { term: 'Outer product', def: 'The matrix v kᵀ made from two vectors: entry (i, j) is v[i]·k[j]. Adding it to a matrix M makes M·k point towards v.', chapter },
  'hebbian-learning': { term: 'Hebbian learning', def: '“Cells that fire together wire together”: strengthen a connection in proportion to the activity on both ends. A local rule; it needs no error signal from elsewhere.', chapter },
  'linear-attention': { term: 'Linear attention', def: 'Attention without the softmax, so all past keys and values can be summed into one fixed-size matrix. Cost per token stays constant, but recall is less exact.', chapter },
  'delta-rule': { term: 'Delta rule', def: 'A memory write that adds only the error: M += β (v − M k) kᵀ. It replaces the old value for a key instead of piling on top of it. One gradient step on ‖M k − v‖².', chapter },
  'state-space-model': { term: 'State-space model (SSM)', def: 'A sequence model that carries a fixed-size state from token to token, updated by a learned (often input-dependent) rule. Mamba is the best-known example.', chapter },
  'hybrid-model': { term: 'Hybrid model', def: 'A model that mixes cheap fixed-state layers (linear attention, DeltaNet, Mamba) with a few full attention layers, for long context at lower cost.', chapter },
  'test-time-training': { term: 'Test-time training (TTT)', def: 'Updating some weights with gradient steps on a self-supervised loss while the model is being used, on the input it is processing.', chapter },
  'surprise-signal': { term: 'Surprise (Titans)', def: 'How wrong a test-time memory is about the current input: the gradient of its prediction loss. Large surprise means a large write.', chapter },
  titans: { term: 'Titans', def: 'A 2025 Google architecture that adds a neural long-term memory, an MLP updated at test time by surprise gradients with momentum and forgetting, next to attention.', chapter },
  'model-editing': { term: 'Model editing', def: 'Changing one specific fact or behaviour in a trained model by directly modifying a few weights, without retraining.', chapter },
  'causal-tracing': { term: 'Causal tracing', def: 'Corrupting the input, then restoring one hidden state at a time to see which layer and position bring the right answer back. Used by ROME to locate facts.', chapter },
  rome: { term: 'ROME', def: 'Rank-One Model Editing (2022): changes one fact by adding a rank-one (outer-product) update to one MLP weight matrix in a middle layer.', chapter },
  memit: { term: 'MEMIT', def: 'Mass-Editing Memory in a Transformer (2022): extends ROME to thousands of facts at once, spreading the update over several MLP layers.', chapter },
  'ripple-effect': { term: 'Ripple effect', def: 'The facts that should change as a consequence of an edit (new capital ⇒ new answers to related questions). Edited models often miss them.', chapter },
  hypernetwork: { term: 'Hypernetwork', def: 'A network whose output is the weights (or a weight update, such as a LoRA) of another network.', chapter },
  'credit-assignment': { term: 'Credit assignment', def: 'Working out which internal weights caused an outcome, and by how much. Backpropagation solves it exactly but needs stored activations and a backward pass.', chapter },
  'forward-forward': { term: 'Forward-forward algorithm', def: 'Hinton’s 2022 alternative to backprop: each layer learns locally to give high “goodness” to real data and low to negative data, using two forward passes and no backward pass.', chapter },
}

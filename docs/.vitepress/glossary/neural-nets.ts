import type { GlossaryEntry } from '../glossary'

const chapter = '/foundations/neural-nets'

export const terms: Record<string, GlossaryEntry> = {
  neuron: {
    term: 'Neuron',
    def: 'The basic unit of a neural network: multiply each input by a weight, add them up plus a bias, then apply an activation function. Outputs one number.',
    chapter,
  },
  weight: {
    term: 'Weight',
    def: 'A number that says how strongly one input counts towards a neuron’s sum. Weights are set by training, not written by hand.',
    chapter,
  },
  bias: {
    term: 'Bias',
    def: 'A number a neuron adds to its weighted sum. It shifts the threshold at which the neuron becomes active.',
    chapter,
  },
  parameter: {
    term: 'Parameter',
    def: 'Any number in the model that training sets: all the weights and biases. “A 7B model” has about 7 billion of them.',
    chapter,
  },
  'activation-function': {
    term: 'Activation function',
    def: 'A simple nonlinear function applied to each neuron’s sum, such as ReLU or GELU. Without it, stacked layers collapse into one linear layer.',
    chapter,
  },
  relu: {
    term: 'ReLU',
    def: 'Rectified linear unit: max(0, z). Passes positive values through and turns negative values into zero. The simplest common activation function.',
    chapter,
  },
  gelu: {
    term: 'GELU',
    def: 'Gaussian error linear unit: a smooth version of ReLU, used in GPT-2 and BERT. Negative inputs are scaled towards zero instead of cut off sharply.',
    chapter,
  },
  layer: {
    term: 'Layer',
    def: 'A group of neurons that all read the same inputs. A layer is one matrix multiply plus a bias, usually followed by an activation function.',
    chapter,
  },
  mlp: {
    term: 'MLP (multi-layer perceptron)',
    def: 'A stack of fully connected layers with activation functions between them. Transformers contain an MLP block in every layer.',
    chapter,
  },
  activations: {
    term: 'Activations',
    def: 'The intermediate numbers a network computes for one input as it runs. Unlike weights, they are recomputed for every input and then thrown away.',
    chapter,
  },
  'forward-pass': {
    term: 'Forward pass',
    def: 'Running the network on an input: apply each layer in order, from input to output. Using a trained model is only forward passes.',
    chapter,
  },
  tensor: {
    term: 'Tensor',
    def: 'An n-dimensional array of numbers with a shape, such as a 4096 × 11008 matrix. Model files are collections of named tensors.',
    chapter,
  },
  checkpoint: {
    term: 'Checkpoint',
    def: 'A saved file of a model’s parameters (its named tensors). Together with the architecture code, it is the whole model.',
    chapter,
  },
  'universal-approximation': {
    term: 'Universal approximation theorem',
    def: 'A network with one hidden layer and enough neurons can approximate any continuous function on a bounded region as closely as you like. It says nothing about how many neurons, or how to find the weights.',
    chapter,
  },
}

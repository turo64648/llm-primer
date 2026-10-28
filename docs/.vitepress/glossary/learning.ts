import type { GlossaryEntry } from '../glossary'

const chapter = '/foundations/learning'

export const terms: Record<string, GlossaryEntry> = {
  loss: {
    term: 'Loss',
    def: 'One number that measures how wrong the model is on some data. Training changes the weights to make it smaller. Also called the loss function or objective.',
    chapter,
  },
  mse: {
    term: 'Mean squared error (MSE)',
    def: 'A loss for predicting numbers: the average of (prediction − target)². Large errors count much more than small ones.',
    chapter,
  },
  'loss-landscape': {
    term: 'Loss landscape',
    def: 'The loss viewed as a surface over all possible weight settings. Training is a walk downhill on it, one step at a time.',
    chapter,
  },
  'gradient-descent': {
    term: 'Gradient descent',
    def: 'The basic training loop: compute the gradient of the loss, move every weight a small step against it, repeat.',
    chapter,
  },
  'learning-rate': {
    term: 'Learning rate',
    def: 'The step size in gradient descent: how far to move the weights per unit of gradient. Too small is slow; too large makes training diverge.',
    chapter,
  },
  backpropagation: {
    term: 'Backpropagation',
    def: 'Computing the gradient of the loss for every parameter by applying the chain rule backwards through the computation graph, from the loss to the inputs. Also called the backward pass.',
    chapter,
  },
  'computation-graph': {
    term: 'Computation graph',
    def: 'The record of every operation in a forward pass and which values it used. Backpropagation walks it in reverse.',
    chapter,
  },
  autograd: {
    term: 'Autograd (automatic differentiation)',
    def: 'Software that records the computation graph during the forward pass and computes exact gradients from it automatically. PyTorch’s loss.backward() is autograd.',
    chapter,
  },
  minibatch: {
    term: 'Minibatch',
    def: 'A small random subset of the training data used for one gradient step. Its gradient is a noisy estimate of the gradient over all the data.',
    chapter,
  },
  sgd: {
    term: 'Stochastic gradient descent (SGD)',
    def: 'Gradient descent where each step uses the gradient of a random minibatch instead of the whole dataset. Cheaper steps, with noise.',
    chapter,
  },
  epoch: {
    term: 'Epoch',
    def: 'One full pass over the training data. LLM pretraining often sees most of its data only about once.',
    chapter,
  },
  momentum: {
    term: 'Momentum',
    def: 'An optimizer trick: step along a running average of recent gradients instead of the latest one. Noise cancels out and consistent directions build up speed.',
    chapter,
  },
  'weight-decay': {
    term: 'Weight decay',
    def: 'Shrinking every weight a little towards zero at each step. A regulariser that discourages large weights. AdamW applies it directly to the weights.',
    chapter,
  },
  adamw: {
    term: 'AdamW',
    def: 'Adam with weight decay applied directly to the weights instead of added to the loss. The default optimizer for training LLMs.',
    chapter,
  },
  'lr-schedule': {
    term: 'Learning rate schedule',
    def: 'A plan for changing the learning rate during training. The common LLM recipe: a short linear warmup from near zero, then a slow cosine-shaped decay.',
    chapter,
  },
  'vanishing-gradients': {
    term: 'Vanishing and exploding gradients',
    def: 'In a deep network the gradient is a product of many per-layer factors. If they are mostly below 1 it shrinks towards zero; above 1, it blows up.',
    chapter,
  },
  overfitting: {
    term: 'Overfitting',
    def: 'Fitting the training data so closely, noise included, that performance on new data gets worse. Shows up as training loss falling while validation loss rises.',
    chapter,
  },
  'validation-set': {
    term: 'Validation set',
    def: 'Data held out from training and used only to measure how well the model does on examples it has not seen.',
    chapter,
  },
  regularization: {
    term: 'Regularisation',
    def: 'Anything that makes a model generalise better at some cost to how well it fits the training data: weight decay, dropout, early stopping, more data.',
    chapter,
  },
  'double-descent': {
    term: 'Double descent',
    def: 'The observation that test error can rise and then fall again as models grow past the size where they fit the training data exactly. Its explanation is still researched.',
    chapter,
  },
}

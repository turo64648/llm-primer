// Flashcards for the How Networks Learn chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'What is a loss, and why must it be a single number?',
    a: 'A number that measures how wrong the network is on some data. Training needs one number so it can ask, for every weight, “which way makes this smaller?”',
  },
  {
    q: 'Why use cross-entropy rather than MSE for classification?',
    a: 'Cross-entropy, −log(p of the right answer), punishes confident mistakes hard, and its gradient on the logits is p − onehot. With MSE on probabilities, a confidently wrong prediction gets a tiny gradient, so the worst examples learn slowest.',
  },
  {
    q: 'What is the loss for a language model?',
    a: 'Cross-entropy on the next token: at every position, −log of the probability the model gave to the token that actually comes next, averaged over all positions.',
  },
  {
    q: 'Why does gradient descent step against the gradient?',
    a: 'The gradient points in the direction in which the loss rises fastest. Moving a small step the opposite way lowers the loss fastest, for small enough steps.',
  },
  {
    q: 'What goes wrong if the learning rate is too large or too small?',
    a: 'Too large: steps overshoot the valley and the loss grows or becomes NaN. Too small: progress is so slow that the compute is wasted.',
  },
  {
    q: 'What does backpropagation actually do?',
    a: 'It walks the computation graph from the loss back to the weights, multiplying each node’s gradient by its local derivatives and adding the results into its inputs. It computes the exact gradient; it does not decide the update.',
  },
  {
    q: 'Why reverse mode rather than forward mode for training?',
    a: 'Training has one output (the loss) and billions of inputs (the weights). One backward pass gives the derivative of that one output with respect to every input. Forward mode would need one pass per weight.',
  },
  {
    q: 'Why does training need much more activation memory than inference?',
    a: 'The backward pass needs the intermediate values of the forward pass, so they must all be kept until it runs. Inference can drop each layer’s output once the next layer has used it.',
  },
  {
    q: 'How do you test a hand-written gradient?',
    a: 'Gradient check: nudge one weight up and down by a tiny ε, compute (L(w+ε) − L(w−ε)) / 2ε, and compare with the backward pass. Too slow to train with, but a reliable test.',
  },
  {
    q: 'Why train on minibatches instead of the whole dataset?',
    a: 'An exact gradient would need a pass over all the data per step. A random minibatch gives a noisy estimate that is right on average for a tiny fraction of the cost, so you get many more steps.',
  },
  {
    q: 'How does momentum help?',
    a: 'It steps along a running average of recent gradients. Directions that flip from step to step cancel; directions that stay consistent build up speed, so it stops zigzagging across narrow valleys.',
  },
  {
    q: 'What does Adam store per parameter, and why?',
    a: 'Two running averages: of the gradient (momentum) and of the squared gradient (typical size). Dividing one by the square root of the other gives each weight a step of about the learning rate, whatever its gradient scale. 8 bytes per parameter in fp32.',
  },
  {
    q: 'Why AdamW instead of Adam plus an L2 penalty in the loss?',
    a: 'An L2 penalty’s gradient goes through Adam’s per-weight scaling, so weights with large gradients get almost no decay. AdamW shrinks the weights directly, separately from the adaptive step.',
  },
  {
    q: 'About how many bytes per parameter does mixed-precision AdamW training need, versus bf16 inference?',
    a: 'About 16 (2 weights + 2 gradient + 4 fp32 master copy + 8 Adam state), plus activations. Inference needs about 2. So a 7B model is about 14 GB to serve and 112 GB to train.',
  },
  {
    q: 'Why warm up the learning rate, then decay it?',
    a: 'Early on, weights are random and Adam’s averages are empty, so big steps are dangerous. Later, large steps make fast progress, and small final steps let the weights settle into a valley instead of bouncing around it.',
  },
  {
    q: 'Why do gradients vanish or explode in deep networks?',
    a: 'The gradient for an early layer is a product of one factor per later layer. Factors mostly below 1 shrink it exponentially; above 1, it blows up. Scaled initialisation, ReLU, residual connections, normalisation and clipping keep it in range.',
  },
  {
    q: 'How do you spot overfitting?',
    a: 'Training loss keeps falling while loss on a held-out validation set rises. The network is memorising its training examples, noise included.',
  },
  {
    q: 'What is double descent, and why is it a puzzle?',
    a: 'Test error rises as a model approaches the size where it fits the training data exactly, then falls again as it grows further. Classical theory predicts only the rise. Why big networks generalise this well is still an active research question.',
  },
]

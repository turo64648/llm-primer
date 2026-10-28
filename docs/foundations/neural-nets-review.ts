// Flashcards for the Neural Networks from Scratch chapter. Strings may contain inline HTML.

export const cards = [
  {
    q: 'What does a single neuron compute?',
    a: 'It multiplies each input by a weight, adds them up plus a bias, and passes the result through an activation function such as ReLU. The output is one number.',
  },
  {
    q: 'What does the bias do?',
    a: 'It shifts the weighted sum up or down, which moves the threshold where the neuron turns on. In the AND neuron, a bias of −1.5 means both inputs must be on.',
  },
  {
    q: 'Why is a layer a matrix multiply?',
    a: 'A layer is many neurons reading the same input. Stack each neuron’s weights as one row of a matrix W; then all their dot products together are W·x.',
  },
  {
    q: 'Why do stacked linear layers collapse into one?',
    a: 'Matrix multiplication is associative: B(Ax) = (BA)x, and BA is a single matrix. Biases only add one combined bias. So without a nonlinearity, depth adds nothing.',
  },
  {
    q: 'Why can no linear model compute XOR?',
    a: 'A linear model splits the inputs with a straight line. XOR’s two “1” cases sit on opposite corners of the square, so no line separates them from the two “0” cases.',
  },
  {
    q: 'How does ReLU make a network more expressive?',
    a: 'ReLU(z) = max(0, z) adds one kink per neuron. Many neurons with kinks in different places can build any bent shape, like a polyline from straight segments.',
  },
  {
    q: 'What is the difference between the activation function and the activations?',
    a: 'The activation function is the fixed bend (ReLU, GELU). The activations are the intermediate values a forward pass computes for one input, which are thrown away afterwards.',
  },
  {
    q: 'How many parameters does a layer from n_in to n_out inputs have?',
    a: 'n_in × n_out weights plus n_out biases (if it uses biases). A 4096 → 11008 layer has about 45 million.',
  },
  {
    q: 'What does the universal approximation theorem guarantee, and what does it not?',
    a: 'One hidden layer with enough neurons can approximate any continuous function on a bounded region. It does not say how many neurons, how to find the weights, or whether the result generalises.',
  },
  {
    q: 'Why use deep networks if one hidden layer is already universal?',
    a: 'Some functions need vastly more neurons in one wide layer than in several narrower ones. Depth lets each layer build on features found by the layer before.',
  },
  {
    q: 'What is a model checkpoint, physically?',
    a: 'A file of named tensors (arrays with shapes) holding the weights. Together with a short, fixed program that says how to multiply them, it is the whole model.',
  },
  {
    q: 'How big is a 7B-parameter model in bf16, and why?',
    a: 'About 14 GB: 7 billion parameters × 2 bytes each. In float32 it would be about 28 GB; in 4-bit about 3.5 GB.',
  },
  {
    q: 'Roughly how much compute does one token cost in a forward pass?',
    a: 'About 2 × parameters floating-point operations, because each weight is used in one multiply and one add. For a 7B model, about 14 billion per token.',
  },
  {
    q: 'Why is generating one token for one user limited by memory bandwidth?',
    a: 'Each token needs every weight read once, so a 7B bf16 model reads about 14 GB per token. At about 3 TB/s that is about 5 ms, while the arithmetic itself is much faster.',
  },
  {
    q: 'Does using a model change its weights?',
    a: 'No. A forward pass only reads the weights and writes activations. Lasting changes need a training run that produces a new checkpoint.',
  },
  {
    q: 'How does an LLM relate to the MLPs in this chapter?',
    a: 'Each of its repeated blocks contains an MLP block (widen, activation function, narrow) next to attention. In Llama 2 7B, MLP blocks hold about two-thirds of the parameters.',
  },
]

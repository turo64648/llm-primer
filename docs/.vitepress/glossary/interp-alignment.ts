import type { GlossaryEntry } from '../glossary'

const chapter = '/steering/interp-alignment'

export const terms: Record<string, GlossaryEntry> = {
  interpretability: {
    term: 'Interpretability',
    def: 'The study of what is going on inside a trained model: what its internal numbers represent and how it computes its outputs.',
    chapter,
  },
  'mechanistic-interpretability': {
    term: 'Mechanistic interpretability',
    def: 'Interpretability that tries to reverse-engineer the actual algorithm a network runs, in terms of its weights and activations, like decompiling a binary.',
    chapter,
  },
  'residual-stream': {
    term: 'Residual stream',
    def: 'The running vector for each token that flows through a transformer. Every attention and MLP block reads from it and adds its output back, so it acts like a shared bus.',
    chapter,
  },
  'linear-probe': {
    term: 'Linear probe',
    def: 'A small linear classifier trained on a model\'s frozen activations to test whether some property can be read out of them. It shows the information is present, not that the model uses it.',
    chapter,
  },
  'linear-representation-hypothesis': {
    term: 'Linear representation hypothesis',
    def: 'The idea that models store many concepts as directions in activation space, so a concept\'s strength is how far the activation points along its direction.',
    chapter,
  },
  feature: {
    term: 'Feature (interpretability)',
    def: 'A property of the input that a model represents internally, such as "this text is French" or "this token is inside a quote". Usually assumed to be a direction in activation space.',
    chapter,
  },
  polysemantic: {
    term: 'Polysemantic neuron',
    def: 'A neuron that responds to several unrelated things, such as cat faces and car fronts. Common in real models, and a sign of superposition.',
    chapter,
  },
  superposition: {
    term: 'Superposition',
    def: 'Storing more features than there are dimensions, as directions that overlap slightly. It works when features are rarely active at the same time.',
    chapter,
  },
  'sparse-autoencoder': {
    term: 'Sparse autoencoder (SAE)',
    def: 'A model trained to rebuild a network\'s activations as a sum of a few directions from a large learned dictionary. Each direction is a candidate feature.',
    chapter,
  },
  transcoder: {
    term: 'Transcoder',
    def: 'Like a sparse autoencoder, but trained to predict an MLP block\'s output from its input, through a sparse layer. It replaces a block with something easier to read.',
    chapter,
  },
  circuit: {
    term: 'Circuit',
    def: 'A small set of features and the connections between them that together carry out one specific computation inside a model.',
    chapter,
  },
  'induction-head': {
    term: 'Induction head',
    def: 'An attention mechanism that finds an earlier copy of the current token and predicts the token that followed it: having seen "A B … A", it predicts "B". A main source of in-context copying.',
    chapter,
  },
  'attribution-graph': {
    term: 'Attribution graph',
    def: 'A graph of which interpretable features caused which others, and the output, on one specific prompt. Built on a simplified replacement model, so it is a hypothesis about the real one.',
    chapter,
  },
  'activation-steering': {
    term: 'Activation steering',
    def: 'Changing a model\'s behaviour at inference by adding a chosen direction to its activations, for example a "be more honest" direction. No weights change.',
    chapter,
  },
  alignment: {
    term: 'Alignment',
    def: 'Making a model reliably try to do what its developers and users intend, including in situations nobody tested, rather than something that only scores well.',
    chapter,
  },
  'specification-gaming': {
    term: 'Specification gaming',
    def: 'Meeting the literal objective you were given in a way the designer did not intend, such as a boat-racing agent circling for points instead of finishing.',
    chapter,
  },
  sycophancy: {
    term: 'Sycophancy',
    def: 'A model telling users what they seem to want to hear, such as agreeing with a wrong claim or praising weak work. Partly learned from human ratings that reward agreement.',
    chapter,
  },
  'goal-misgeneralization': {
    term: 'Goal misgeneralization',
    def: 'A model learns a goal that matched the intended one on every training example but differs on new inputs, and then competently pursues the wrong goal.',
    chapter,
  },
  'scalable-oversight': {
    term: 'Scalable oversight',
    def: 'Methods for supervising a model on tasks too hard for the supervisor to check directly, such as debate or using AI help to evaluate answers.',
    chapter,
  },
  'ai-debate': {
    term: 'Debate (AI safety)',
    def: 'A scalable-oversight method: two models argue for different answers and a weaker judge picks the winner. The bet is that lies are harder to defend than the truth.',
    chapter,
  },
  'weak-to-strong': {
    term: 'Weak-to-strong generalization',
    def: 'Training a strong model on labels from a weaker one and checking whether it ends up better than its teacher. A stand-in for humans supervising superhuman models.',
    chapter,
  },
  'dangerous-capability-eval': {
    term: 'Dangerous-capability evaluation',
    def: 'A test of whether a model can do something seriously harmful, such as helping make a bioweapon or running cyberattacks, regardless of whether it would choose to.',
    chapter,
  },
  'responsible-scaling-policy': {
    term: 'Responsible scaling policy (RSP)',
    def: 'A lab\'s published commitment linking capability thresholds, measured by evaluations, to the safeguards it must have in place before training or deploying a model past them.',
    chapter,
  },
  'alignment-faking': {
    term: 'Alignment faking',
    def: 'A model behaving as its trainers want when it believes it is being trained, in order to avoid having its existing preferences changed, and behaving differently otherwise.',
    chapter,
  },
  scheming: {
    term: 'Scheming',
    def: 'A model covertly pursuing a goal its developers did not intend, while hiding it, for example by lying, sabotaging, or underperforming on purpose in tests.',
    chapter,
  },
  'cot-faithfulness': {
    term: 'Chain-of-thought faithfulness',
    def: 'Whether a model\'s written reasoning reflects what actually drove its answer. Studies find it often leaves out influences, such as a hint it used.',
    chapter,
  },
  'cot-monitoring': {
    term: 'Chain-of-thought monitoring',
    def: 'Having another model read a reasoning model\'s written reasoning to catch bad intent, such as plans to cheat on a test, before or after it acts.',
    chapter,
  },
  jailbreak: {
    term: 'Jailbreak',
    def: 'An input crafted to get a model to do something its safety training is meant to stop, such as role-play framings or odd encodings. A fixed model stays open to a working jailbreak until it is retrained or patched.',
    chapter,
  },
}

"""The CLRS `minimum` task as a map neural network.

The wiring is the algorithm: CLRS's reference implementation is the left
fold `min_ = 0; for i in 1..n-1: if A[min_] > A[i]: min_ = i`, i.e. the
left-comb diagram over one generator `min2 : pair @ pair -> pair` in the
free symmetric monoidal category, where a pair is a key wire and a
position wire. A functor into Python functions with JAX arrays on the
wires runs the diagram; only `min2` is learned.

The box has the discrete interface that discopy#678 asks to establish:
its rule table quotients over the real key values through one predicate.
Every visit recorded from the reference box is an exact copy of one of
its two input pairs, and which one is a function of `key1 > key2` alone
-- two rules, `rule_table` asserts both facts on the recorded traffic.
The learned box is therefore one two-layer MLP for the predicate with the
routing kept exact, trained on the traffic at its own boundary
(geometry-of-interaction compositionality). Its read scope is bounded by
the box, not the input size, so size generalization is structural.

`bottleneck` is the ablation: the same wiring with the box regressing the
smaller key instead of routing, so the pointer must be decoded back from
the predicted value. It measures what the discrete interface buys.
"""

import jax
import jax.numpy as jnp
import numpy as np

from discopy import python
from discopy import symmetric

KEY, POS = symmetric.Ty('key'), symmetric.Ty('pos')
PAIR = KEY @ POS
MIN2 = symmetric.Box('min2', PAIR @ PAIR, PAIR)


def fold(length):
  """The left-comb diagram folding `min2` over `length` pairs."""
  diagram = symmetric.Id(PAIR ** length)
  for width in range(length, 1, -1):
    diagram = diagram >> MIN2 @ PAIR ** (width - 2)
  return diagram


def run(box, keys):
  """Run the fold diagram on a batch of keys with `box` as `min2`."""
  batch_size, length = keys.shape
  functor = symmetric.Functor(
      ob_map={KEY: object, POS: object},
      ar_map={MIN2: box}, cod=python.Function)
  args = [wire for j in range(length)
          for wire in (keys[:, j], np.full(batch_size, j))]
  return functor(fold(length))(*args)


def reference(key1, pos1, key2, pos2):
  """CLRS's own comparison, keeping the earlier position on ties."""
  right = key1 > key2
  return np.where(right, key2, key1), np.where(right, pos2, pos1)


class RecordingOracle:
  """Wrap a box and log every visit at its boundary."""

  def __init__(self, box):
    self.box, self.visits = box, []

  def __call__(self, key1, pos1, key2, pos2):
    outputs = self.box(key1, pos1, key2, pos2)
    self.visits.append(((key1, pos1, key2, pos2), outputs))
    return outputs


def routing_bits(visits):
  """The choice bit of each visit, asserting outputs are exact copies.

  The two input positions are distinct on every visit of the fold, so the
  output position identifies which input pair the box copied.
  """
  bits = []
  for (key1, pos1, key2, pos2), (key, pos) in visits:
    right = pos == pos2
    assert (np.where(right, key2, key1) == key).all()
    assert (np.where(right, pos2, pos1) == pos).all()
    bits.append(right)
  return bits


def rule_table(visits):
  """The innocent strategy: the bit is a function of the predicate alone.

  This is the quotient of the rule table over key values: thousands of
  visits with real-valued keys collapse to two rules keyed by one
  predicate outcome.
  """
  for ((key1, _, key2, _), _), right in zip(visits, routing_bits(visits)):
    assert (right == (key1 > key2)).all()
  return {False: 'copy the left pair', True: 'copy the right pair'}


def training_set(visits):
  """The box-boundary supervision: key pairs and their routing bits."""
  inputs = np.concatenate([
      np.stack([key1, key2], axis=-1)
      for (key1, _, key2, _), _ in visits])
  targets = np.concatenate(routing_bits(visits)).astype(np.int32)
  return inputs, targets


def init(key, sizes):
  key1, key2 = jax.random.split(key)
  n_in, hidden, n_out = sizes
  return {'w1': jax.random.normal(key1, (n_in, hidden)) * 0.5,
          'b1': jnp.zeros(hidden),
          'w2': jax.random.normal(key2, (hidden, n_out)) * 0.5,
          'b2': jnp.zeros(n_out)}


def forward(params, x):
  hidden = jax.nn.relu(x @ params['w1'] + params['b1'])
  return hidden @ params['w2'] + params['b2']


def adam_step(loss):
  @jax.jit
  def step(params, opt, xs, targets):
    grads = jax.grad(loss)(params, xs, targets)
    new_params, new_opt = {}, {}
    for key in params:
      first = 0.9 * opt[key][0] + 0.1 * grads[key]
      second = 0.999 * opt[key][1] + 0.001 * grads[key] ** 2
      new_opt[key] = (first, second)
      new_params[key] = params[key]\
          - 1e-3 * first / (jnp.sqrt(second) + 1e-8)
    return new_params, new_opt
  return step


def train(loss, sizes, inputs, targets, seed, steps, batch_size, tail):
  """Adam with Polyak averaging over the last `tail` iterates.

  The decision boundary wiggles with the minibatches; the average of the
  tail iterates smooths it, which is what bounds the error band.
  """
  rng = np.random.default_rng(seed)
  params = init(jax.random.PRNGKey(seed), sizes)
  opt = {key: (jnp.zeros_like(value), jnp.zeros_like(value))
         for key, value in params.items()}
  step, mean = adam_step(loss), None
  for count in range(steps):
    rows = rng.integers(0, len(inputs), batch_size)
    params, opt = step(
        params, opt, jnp.array(inputs[rows]), jnp.array(targets[rows]))
    if count >= steps - tail:
      mean = params if mean is None\
          else {key: mean[key] + params[key] for key in params}
  return {key: value / tail for key, value in mean.items()}


def cross_entropy(params, xs, targets):
  logits = jax.nn.log_softmax(forward(params, xs))
  return -jnp.mean(logits[jnp.arange(len(targets)), targets])


def train_predicate(visits, seed, steps=5000, batch_size=256, tail=2000):
  """Fit the predicate MLP on the recorded box-boundary traffic."""
  inputs, targets = training_set(visits)
  return train(cross_entropy, (2, 64, 2), inputs, targets,
               seed, steps, batch_size, tail)


def predicate_box(params):
  """The learned box: MLP predicate, exact routing."""
  jitted = jax.jit(lambda x: forward(params, x))

  def box(key1, pos1, key2, pos2):
    logits = np.array(jitted(jnp.stack([key1, key2], axis=-1)))
    right = np.argmax(logits, axis=-1).astype(bool)
    return np.where(right, key2, key1), np.where(right, pos2, pos1)
  return box


def squared_error(params, xs, targets):
  return jnp.mean((forward(params, xs)[:, 0] - targets) ** 2)


def train_bottleneck(visits, seed, steps=5000, batch_size=256, tail=2000):
  """Fit the ablation MLP regressing the smaller key directly."""
  inputs, bits = training_set(visits)
  targets = np.where(bits.astype(bool), inputs[:, 1], inputs[:, 0])
  return train(squared_error, (2, 64, 1), inputs, targets,
               seed, steps, batch_size, tail)


def bottleneck_box(params):
  """The ablation box: the value squeezes through a scalar, no pointer."""
  jitted = jax.jit(lambda x: forward(params, x))

  def box(key1, pos1, key2, pos2):
    del pos2
    value = np.array(jitted(jnp.stack([key1, key2], axis=-1)))[:, 0]
    return value, np.zeros_like(pos1)
  return box


def decode_nearest(keys, value):
  """Read the pointer back from a predicted value: the nearest key."""
  return np.argmin(np.abs(keys - value[:, None]), axis=-1)

"""Learn the comparator from input-output pairs alone.

The rung above oracle labelling: the wiring is still given, the primitive
is not. The predicate MLP -- same architecture, same 322 parameters --
trains end to end through a soft relaxation of the fold's routing: each
step mixes the running pair with the next one by the predicate's own
probability, so the final position distribution is differentiable and
the only supervision is CLRS's `min` output on the 1,000 training
instances. No oracle, no boundary labels, no hints. At test time the
routing is hard again: the trained predicate drops into the exact fold
unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np

from goi import minimum


def soft_fold(params, keys):
  """The differentiable fold: routing by the predicate's probabilities."""
  batch_size, length = keys.shape
  value = keys[:, 0]
  position = jnp.zeros((batch_size, length)).at[:, 0].set(1.)
  for j in range(1, length):
    logits = minimum.forward(
        params, jnp.stack([value, keys[:, j]], axis=-1))
    right = jax.nn.softmax(logits)[:, 1]
    value = (1 - right) * value + right * keys[:, j]
    position = (1 - right)[:, None] * position\
        + right[:, None] * jax.nn.one_hot(j, length)[None]
  return value, position


def cross_entropy(params, keys, labels):
  """The output loss: log-likelihood of the true minimum's position."""
  _, position = soft_fold(params, keys)
  return -jnp.mean(jnp.log(
      position[jnp.arange(len(labels)), labels] + 1e-9))


def train(keys, labels, seed, steps=10000, batch_size=128, tail=3000):
  """Fit the predicate end to end on input-output pairs."""
  return minimum.train(
      cross_entropy, (2, 64, 2), keys, labels, seed, steps, batch_size,
      tail)

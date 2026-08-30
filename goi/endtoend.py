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


def soft_network(params, keys):
  """The differentiable transposition network: soft compare-exchanges.

  Returns the position distribution of each output wire, so the loss can
  ask that wire `w` carry the `w`-th smallest key's original position.
  """
  batch_size, length = keys.shape
  values = keys
  position = jnp.tile(jnp.eye(length)[None], (batch_size, 1, 1))
  for step in range(length):
    index = jnp.arange(step % 2, length - 1, 2)
    left, right_ = values[:, index], values[:, index + 1]
    logits = minimum.forward(params, jnp.stack(
        [left, right_, left - right_], axis=-1).reshape(-1, 3))
    swap = jax.nn.softmax(logits)[:, 1].reshape(batch_size, len(index))
    values = values\
        .at[:, index].set((1 - swap) * left + swap * right_)\
        .at[:, index + 1].set((1 - swap) * right_ + swap * left)
    rows_left, rows_right = position[:, index], position[:, index + 1]
    position = position\
        .at[:, index].set((1 - swap[..., None]) * rows_left
                          + swap[..., None] * rows_right)\
        .at[:, index + 1].set((1 - swap[..., None]) * rows_right
                              + swap[..., None] * rows_left)
  return values, position


def sort_cross_entropy(params, keys, sorted_pos):
  """The output loss: log-likelihood of the true sorted order."""
  _, position = soft_network(params, keys)
  batch_size, length = keys.shape
  rows = position[
      jnp.arange(batch_size)[:, None], jnp.arange(length)[None], sorted_pos]
  return -jnp.mean(jnp.log(rows + 1e-9))


def train_sort(keys, sorted_pos, seed, steps=10000, batch_size=128,
               tail=3000):
  """Fit the exchange predicate end to end on input-output pairs."""
  return minimum.train(
      sort_cross_entropy, (3, 256, 2), keys, sorted_pos, seed, steps,
      batch_size, tail)

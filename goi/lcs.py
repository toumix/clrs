"""The CLRS `lcs_length` task as a map neural network.

The wiring is the DP grid, and the whole interface of its one cell is
bits: neighbouring LCS counts differ by at most one, so the cell reads
`(match, up_delta, left_delta)` -- the count differences to its diagonal
neighbour -- and decides the direction `b`, from which the outgoing
deltas are exact arithmetic. Zero-padding the grid makes CLRS's
special-cased first row and column exact instances of the same rule, so
the task is one cell box (at most eight rules) and one character
equality box (sixteen rules on the four-letter alphabet), both finite:
size generalization is structural, and no wire ever carries a quantity
that grows with the input.

Everything is learned end to end from CLRS's `b` output alone: soft
routing through the grid during training, hard routing in the exact
recurrence at test.
"""

import jax
import jax.numpy as jnp
import numpy as np

from goi import minimum


def split_strings(feedback):
  """The two strings as integer characters, shape [batch, nx], [batch, ny]."""
  mask, chars = None, None
  for data_point in feedback.features.inputs:
    if data_point.name == 'string':
      mask = data_point.data
    if data_point.name == 'key':
      chars = np.argmax(data_point.data, axis=-1)
  assert (mask == mask[:1]).all()
  first = mask[0].astype(bool)
  return chars[:, ~first], chars[:, first]


def reference_b(x, y):
  """The direction table of CLRS's reference, computed in delta form."""
  batch_size, nx = x.shape
  ny = y.shape[1]
  match = x[:, :, None] == y[:, None, :]
  count = np.zeros((batch_size, nx + 1, ny + 1), dtype=int)
  b = np.zeros((batch_size, nx, ny), dtype=int)
  for i in range(nx):
    for j in range(ny):
      up = count[:, i, j + 1] - count[:, i, j]
      left = count[:, i + 1, j] - count[:, i, j]
      b[:, i, j] = np.where(
          match[:, i, j], 0, np.where(up >= left, 1, 2))
      gain = np.where(match[:, i, j], 1, np.maximum(up, left))
      count[:, i + 1, j + 1] = count[:, i, j] + gain
  return b


def cell_visits(x, y):
  """The cell's boundary traffic: (match, up, left) -> b, as flat arrays."""
  batch_size, nx = x.shape
  ny = y.shape[1]
  match = x[:, :, None] == y[:, None, :]
  count = np.zeros((batch_size, nx + 1, ny + 1), dtype=int)
  rows = []
  for i in range(nx):
    for j in range(ny):
      up = count[:, i, j + 1] - count[:, i, j]
      left = count[:, i + 1, j] - count[:, i, j]
      m = match[:, i, j].astype(int)
      b = np.where(m, 0, np.where(up >= left, 1, 2))
      rows.append(np.stack([m, up, left, b], axis=-1))
      gain = np.where(m, 1, np.maximum(up, left))
      count[:, i + 1, j + 1] = count[:, i, j] + gain
  return np.concatenate(rows)


def rule_table(visits):
  """The finite strategy: b is a function of the three bits alone."""
  rules = {}
  for m, up, left, b in visits:
    key = (int(m), int(up), int(left))
    assert rules.setdefault(key, int(b)) == int(b)
  assert len(rules) <= 8
  return rules


def one_hot_pairs(x, y):
  """The equality box's input: both characters one-hot, [.., 8]."""
  return np.concatenate([
      np.eye(4)[x][:, :, None, :].repeat(y.shape[1], axis=2),
      np.eye(4)[y][:, None, :, :].repeat(x.shape[1], axis=1)], axis=-1)


def sub_params(params, prefix):
  return {key[len(prefix):]: value for key, value in params.items()
          if key.startswith(prefix)}


def init(seed):
  """One flat parameter dict for the equality and cell boxes."""
  eq = minimum.init(jax.random.PRNGKey(seed), (8, 32, 2))
  cell = minimum.init(jax.random.PRNGKey(seed + 1), (3, 32, 3))
  return {f"eq_{key}": value for key, value in eq.items()}\
      | {f"cell_{key}": value for key, value in cell.items()}


def soft_grid(params, pairs):
  """The differentiable grid: soft match and direction probabilities."""
  batch_size, nx, ny, _ = pairs.shape
  match = jax.nn.softmax(minimum.forward(
      sub_params(params, "eq_"), pairs))[..., 1]
  logits = []
  up = jnp.zeros((batch_size, ny + 1))
  for i in range(nx):
    left, new_up = jnp.zeros(batch_size), []
    for j in range(ny):
      bits = jnp.stack([match[:, i, j], up[:, j + 1], left], axis=-1)
      cell = minimum.forward(sub_params(params, "cell_"), bits)
      logits.append(cell)
      choice = jax.nn.softmax(cell)
      gain = choice[:, 0] + choice[:, 1] * up[:, j + 1]\
          + choice[:, 2] * left
      new_up.append(gain - left)
      left = gain - up[:, j + 1]
    up = jnp.concatenate(
        [jnp.zeros((batch_size, 1))]
        + [delta[:, None] for delta in new_up], axis=1)
  return jnp.stack(logits, axis=1).reshape(batch_size, nx, ny, 3)


def cross_entropy(params, pairs, b):
  logits = jax.nn.log_softmax(soft_grid(params, pairs))
  return -jnp.mean(jnp.take_along_axis(logits, b[..., None], axis=-1))


def train(feedback, seed, steps=10000, batch_size=32, tail=3000):
  """Fit both boxes end to end on the b outputs alone."""
  x, y = split_strings(feedback)
  pairs, b = one_hot_pairs(x, y), reference_truth_b(feedback)
  rng = np.random.default_rng(seed)
  params = init(seed)
  opt = {key: (jnp.zeros_like(value), jnp.zeros_like(value))
         for key, value in params.items()}
  step, mean = minimum.adam_step(cross_entropy), None
  for count in range(steps):
    rows = rng.integers(0, len(pairs), batch_size)
    params, opt = step(
        params, opt, jnp.array(pairs[rows]), jnp.array(b[rows]))
    if count >= steps - tail:
      mean = params if mean is None\
          else {key: mean[key] + params[key] for key in params}
  return {key: value / tail for key, value in mean.items()}


def reference_truth_b(feedback):
  """The b table read off CLRS's output, shape [batch, nx, ny]."""
  x, y = split_strings(feedback)
  nx = x.shape[1]
  data = feedback.outputs[0].data
  return np.argmax(data[:, :nx, nx:, :3], axis=-1)


def hard_grid(params, x, y):
  """The exact recurrence with the learned boxes and hard routing."""
  jit_eq = jax.jit(lambda v: minimum.forward(sub_params(params, "eq_"), v))
  jit_cell = jax.jit(
      lambda v: minimum.forward(sub_params(params, "cell_"), v))
  pairs = one_hot_pairs(x, y)
  match = np.argmax(np.array(jit_eq(jnp.array(pairs))), axis=-1)
  batch_size, nx = x.shape
  ny = y.shape[1]
  count = np.zeros((batch_size, nx + 1, ny + 1), dtype=int)
  b = np.zeros((batch_size, nx, ny), dtype=int)
  for i in range(nx):
    for j in range(ny):
      up = count[:, i, j + 1] - count[:, i, j]
      left = count[:, i + 1, j] - count[:, i, j]
      bits = np.stack([match[:, i, j], up, left], axis=-1)
      b[:, i, j] = np.argmax(np.array(jit_cell(
          jnp.array(bits, dtype=float))), axis=-1)
      gain = np.choose(b[:, i, j], [np.ones_like(up), up, left])
      count[:, i + 1, j + 1] = count[:, i, j] + gain
  return b


def predictions(feedback, b):
  """The b table as CLRS's edge-categorical output array."""
  batch_size, nx, ny = b.shape
  length = nx + ny
  data = np.zeros((batch_size, length, length, 4), dtype=np.float32)
  batch, i, j = np.meshgrid(
      np.arange(batch_size), np.arange(nx), np.arange(ny), indexing='ij')
  data[batch, i, nx + j, b] = 1.
  return data

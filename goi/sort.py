"""The CLRS `insertion_sort` task as a map neural network.

CLRS scores sorting at the function level: the output is the predecessor
pointer of each node in sorted order, independent of the trace. The
odd-even transposition network computes that function from the same
primitive family as `minimum`: one compare-exchange generator
`sort2 : pair @ pair -> pair @ pair`, applied in `length` alternating
layers of disjoint adjacent pairs. The recorded traffic quotients over
key values exactly as before -- every visit either passes its two pairs
through or exchanges them, and which is a function of `key1 > key2`
alone -- so the single learned component is again the two-layer predicate
MLP, with the exchange routing kept exact.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from discopy import python
from discopy import symmetric

from goi import minimum

SORT2 = symmetric.Box(
    'sort2', minimum.PAIR @ minimum.PAIR, minimum.PAIR @ minimum.PAIR)
RecordingOracle = minimum.RecordingOracle


@lru_cache
def network(length):
  """The odd-even transposition network on `length` pairs."""
  diagram = symmetric.Id(minimum.PAIR ** length)
  for step in range(length):
    layer, offset = symmetric.Id(minimum.PAIR ** (step % 2)), step % 2
    while offset + 1 < length:
      layer = layer @ SORT2
      offset += 2
    if offset > step % 2:
      diagram >>= layer @ minimum.PAIR ** (length - offset)
  return diagram


def run(box, keys):
  """Run the network on a batch of keys; return the sorted positions."""
  functor = symmetric.Functor(
      ob_map={minimum.KEY: object, minimum.POS: object},
      ar_map={SORT2: box}, cod=python.Function)
  wires = minimum.evaluate(
      functor, network(keys.shape[1]), minimum.pair_wires(keys))
  return np.stack(wires[1::2], axis=-1)


def reference(key1, pos1, key2, pos2):
  """CLRS's own comparison as an exchange, stable on ties."""
  right = key1 > key2
  return (np.where(right, key2, key1), np.where(right, pos2, pos1),
          np.where(right, key1, key2), np.where(right, pos1, pos2))


def routing_bits(visits):
  """The exchange bit of each visit, asserting outputs are exact copies."""
  bits = []
  for (key1, pos1, key2, pos2), (low, low_pos, high, high_pos) in visits:
    right = low_pos == pos2
    assert (np.where(right, key2, key1) == low).all()
    assert (np.where(right, pos2, pos1) == low_pos).all()
    assert (np.where(right, key1, key2) == high).all()
    assert (np.where(right, pos1, pos2) == high_pos).all()
    bits.append(right)
  return bits


def rule_table(visits):
  """The innocent strategy: exchange or pass, keyed by the predicate."""
  for ((key1, _, key2, _), _), right in zip(visits, routing_bits(visits)):
    assert (right == (key1 > key2)).all()
  return {False: 'pass the two pairs through', True: 'exchange them'}


def training_set(visits):
  """The box-boundary supervision: the key pairs, their difference and
  their exchange bits. Sorting must resolve every adjacent pair of the
  sorted order, however close, so the predicate reads the difference
  explicitly -- an affine re-encoding of its two keys."""
  inputs = np.concatenate([
      np.stack([key1, key2, key1 - key2], axis=-1)
      for (key1, _, key2, _), _ in visits])
  targets = np.concatenate(routing_bits(visits)).astype(np.int32)
  return inputs, targets


def train_predicate(visits, seed, steps=30000, batch_size=256, tail=10000):
  """Fit the predicate MLP on the recorded box-boundary traffic."""
  inputs, targets = training_set(visits)
  return minimum.train(minimum.cross_entropy, (3, 256, 2), inputs, targets,
                       seed, steps, batch_size, tail)


def predicate_box(params):
  """The learned box: MLP predicate, exact exchange routing."""
  jitted = jax.jit(lambda x: minimum.forward(params, x))

  def box(key1, pos1, key2, pos2):
    logits = np.array(jitted(jnp.stack(
        [key1, key2, key1 - key2], axis=-1)))
    right = np.argmax(logits, axis=-1).astype(bool)
    return (np.where(right, key2, key1), np.where(right, pos2, pos1),
            np.where(right, key1, key2), np.where(right, pos1, pos2))
  return box


def predecessors(sorted_pos):
  """Predecessor pointers from the sorted positions, first to itself."""
  sorted_pos = np.asarray(sorted_pos, dtype=int)
  batch = np.arange(len(sorted_pos))[:, None]
  before = np.concatenate(
      [sorted_pos[:, :1], sorted_pos[:, :-1]], axis=-1)
  pointers = np.empty_like(sorted_pos)
  pointers[batch, sorted_pos] = before
  return pointers

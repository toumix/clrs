"""Train `lcs_length` end to end, test at n = 64, with diagnostics.

The supervision is CLRS's `b` output alone; the trained boxes drop into
the exact recurrence with hard routing. Before scoring, both learned
truth tables are printed against the reference rules — the cell has
eight entries and the equality box sixteen — so a weak score can be told
apart as optimization trouble on identified rows rather than a bug.
Both are read modulo the equality box's polarity, which the output loss
leaves free.

Run with `python -m goi.run_lcs`, or `python -m goi.run_lcs 0` for a
single seed.
"""

import itertools
import time

import jax.numpy as jnp
import numpy as np

from goi import adapter
from goi import lcs
from goi import minimum

SPLITS = {'val (n=16)': (16, 32, 43), 'test (n=64)': (64, 32, 44),
          'far (n=256)': (256, 32, 46)}


def learned_tables(params):
  """The learned cell and equality tables, up to the equality polarity.

  Nothing in the output loss pins which way round the equality box reads:
  a run that learns `not equal` and the correspondingly flipped cell
  rules composes to exactly the same grid and scores exactly the same.
  Comparing both tables against the reference labels literally therefore
  reports a perfect run in that convention as sixteen-out-of-sixteen and
  all-eight wrong, which is the opposite of the truth. Detect the
  polarity the equality box actually settled on, then read the cell
  table in that convention.
  """
  cell = {bits: int(np.argmax(minimum.forward(
      lcs.sub_params(params, 'cell_'), jnp.array([bits], dtype=float))))
      for bits in itertools.product((0, 1), repeat=3)}
  truth = {(m, up, left): 0 if m else (1 if up >= left else 2)
           for m, up, left in itertools.product((0, 1), repeat=3)}
  disagree = sum(
      int(np.argmax(minimum.forward(
          lcs.sub_params(params, 'eq_'),
          jnp.array(np.concatenate([np.eye(4)[a], np.eye(4)[b]])[None])))
          != (a == b))
      for a in range(4) for b in range(4))
  flipped = disagree > 8
  eq_wrong = 16 - disagree if flipped else disagree
  expected = {bits: truth[(1 - bits[0], bits[1], bits[2]) if flipped
                          else bits] for bits in truth}
  wrong = [bits for bits in truth if cell[bits] != expected[bits]]
  return wrong, eq_wrong, flipped


def experiment(seed, steps=30000, batch_size=128, tail=10000):
  start = time.time()
  feedback = adapter.sample('lcs_length', 16, 1000, seed)
  x, y = lcs.split_strings(feedback)
  assert (lcs.reference_b(x, y) == lcs.reference_truth_b(feedback)).all()
  assert len(lcs.rule_table(lcs.cell_visits(x, y))) <= 8
  params = lcs.train(feedback, seed, steps, batch_size, tail)
  cell_wrong, eq_wrong, flipped = learned_tables(params)
  convention = 'not-equal' if flipped else 'equal'
  print(f"seed {seed}: the exact recurrence matches CLRS; equality box "
        f"read as `{convention}`, wrong there: {eq_wrong}/16; cell "
        f"entries wrong: {cell_wrong or 'none'}")
  scores = {}
  for split, (length, batch, sampler_seed) in SPLITS.items():
    f = adapter.sample('lcs_length', length, batch, sampler_seed)
    b = lcs.hard_grid(params, *lcs.split_strings(f))
    scores[split] = adapter.score_categorical(f, 'b', lcs.predictions(f, b))
    print(f"  {split}: {scores[split]:.4f}")
  print(f"  ({time.time() - start:.1f}s)")
  return scores


def main(seeds=(0, 1, 2)):
  results = [experiment(seed) for seed in seeds]
  print(f"\nover seeds {seeds}, mean +- std of 100 * score:")
  for split in SPLITS:
    values = 100 * np.array([result[split] for result in results])
    print(f"  {split}: {values.mean():.2f} +- {values.std():.2f}")


if __name__ == '__main__':
  import sys
  main(tuple(int(seed) for seed in sys.argv[1:]) or (0, 1, 2))

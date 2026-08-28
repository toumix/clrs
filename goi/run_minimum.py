"""Train `minimum` at n = 16, test at n = 64 with CLRS's own metric.

The protocol mirrors the benchmark: 1000 training instances of size 16,
validation on 32 instances of size 16, the out-of-distribution test on 32
instances of size 64 -- plus a wider test batch for a tighter estimate
and a far extrapolation at n = 256, sixteen times the training size.
The published baseline on this task and protocol is Triplet-GMPNN at
96.08% +- 2.07 (Ibarz et al. 2022, table 2).

Run with `python -m goi.run_minimum`.
"""

import time

import numpy as np

from goi import adapter
from goi import minimum

SPLITS = {'val (n=16)': (16, 32, 43), 'test (n=64)': (64, 32, 44),
          'wide test (n=64)': (64, 1000, 45), 'far (n=256)': (256, 32, 46)}
BASELINE = 'Triplet-GMPNN 96.08 +- 2.07 on test (n=64), Ibarz et al. 2022'


def score(box, split, decode=False):
  length, batch_size, seed = SPLITS[split]
  feedback = adapter.sample('minimum', length, batch_size, seed)
  keys = adapter.input_data(feedback, 'key')
  value, pos = minimum.run(box, keys)
  if decode:
    pos = minimum.decode_nearest(keys, value)
  return adapter.score_mask_one(feedback, 'min', pos)


def per_visit_accuracy(box, split):
  """How often the box routes like the reference on the split's traffic."""
  length, batch_size, seed = SPLITS[split]
  keys = adapter.input_data(
      adapter.sample('minimum', length, batch_size, seed), 'key')
  oracle = minimum.RecordingOracle(minimum.reference)
  minimum.run(oracle, keys)
  matches, total = 0, 0
  for (key1, pos1, key2, pos2), (_, pos) in oracle.visits:
    _, learned_pos = box(key1, pos1, key2, pos2)
    matches, total = matches + (learned_pos == pos).sum(), total + pos.size
  return matches / total


def experiment(seed):
  """One seed: record, quotient, train, score. Returns the score dict."""
  start = time.time()
  feedback = adapter.sample('minimum', 16, 1000, seed)
  keys = adapter.input_data(feedback, 'key')
  oracle = minimum.RecordingOracle(minimum.reference)
  _, pos = minimum.run(oracle, keys)
  assert adapter.score_mask_one(feedback, 'min', pos) == 1.0
  n_visits = sum(visit[1][0].size for visit in oracle.visits)
  table = minimum.rule_table(oracle.visits)
  print(f"seed {seed}: the reference box run through the diagram is exact; "
        f"{n_visits} visits collapse to {len(table)} rules")

  predicate = minimum.predicate_box(
      minimum.train_predicate(oracle.visits, seed))
  bottleneck = minimum.bottleneck_box(
      minimum.train_bottleneck(oracle.visits, seed))
  accuracy = per_visit_accuracy(predicate, 'wide test (n=64)')
  print(f"  per-visit accuracy of the predicate on n=64 traffic: "
        f"{accuracy:.6f}")
  scores = {}
  for split in SPLITS:
    scores[split] = (score(predicate, split),
                     score(bottleneck, split, decode=True))
    print(f"  {split}: predicate {scores[split][0]:.4f}, "
          f"value bottleneck {scores[split][1]:.4f}")
  print(f"  ({time.time() - start:.1f}s)")
  return scores


def main(seeds=(0, 1, 2)):
  results = [experiment(seed) for seed in seeds]
  print(f"\nover seeds {seeds}, mean +- std of 100 * score:")
  for split in SPLITS:
    for index, name in enumerate(['predicate', 'value bottleneck']):
      values = 100 * np.array([result[split][index] for result in results])
      print(f"  {split} {name}: {values.mean():.2f} +- {values.std():.2f}")
  print(f"\npublished baseline: {BASELINE}")


if __name__ == '__main__':
  main()

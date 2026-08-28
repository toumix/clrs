"""Train `insertion_sort` at n = 16, test at n = 64 with CLRS's metric.

The protocol mirrors `run_minimum`: 1000 training instances of size 16,
the out-of-distribution test on 32 instances of size 64, a wider test
batch and a far extrapolation at n = 256. The score is CLRS's pointer
metric: the fraction of nodes whose predecessor in sorted order is
predicted exactly.

Run with `python -m goi.run_sort`.
"""

import time

import numpy as np

from goi import adapter
from goi import sort

SPLITS = {'val (n=16)': (16, 32, 43), 'test (n=64)': (64, 32, 44),
          'wide test (n=64)': (64, 1000, 45), 'far (n=256)': (256, 32, 46)}


def score(box, split):
  length, batch_size, seed = SPLITS[split]
  feedback = adapter.sample('insertion_sort', length, batch_size, seed)
  keys = adapter.input_data(feedback, 'key')
  pointers = sort.predecessors(sort.run(box, keys))
  return adapter.score_pointer(feedback, 'pred', pointers)


def experiment(seed):
  """One seed: record, quotient, train, score. Returns the score dict."""
  start = time.time()
  feedback = adapter.sample('insertion_sort', 16, 1000, seed)
  keys = adapter.input_data(feedback, 'key')
  oracle = sort.RecordingOracle(sort.reference)
  pointers = sort.predecessors(sort.run(oracle, keys))
  assert adapter.score_pointer(feedback, 'pred', pointers) == 1.0
  n_visits = sum(visit[1][0].size for visit in oracle.visits)
  table = sort.rule_table(oracle.visits)
  print(f"seed {seed}: the reference box run through the network is exact; "
        f"{n_visits} visits collapse to {len(table)} rules")

  predicate = sort.predicate_box(sort.train_predicate(oracle.visits, seed))
  scores = {}
  for split in SPLITS:
    scores[split] = score(predicate, split)
    print(f"  {split}: predicate {scores[split]:.4f}")
  print(f"  ({time.time() - start:.1f}s)")
  return scores


def main(seeds=(0, 1, 2)):
  results = [experiment(seed) for seed in seeds]
  print(f"\nover seeds {seeds}, mean +- std of 100 * score:")
  for split in SPLITS:
    values = 100 * np.array([result[split] for result in results])
    print(f"  {split}: {values.mean():.2f} +- {values.std():.2f}")


if __name__ == '__main__':
  main()

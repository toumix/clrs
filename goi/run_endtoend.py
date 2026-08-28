"""Train the comparator from input-output pairs alone, test at n = 64.

The protocol is `run_minimum`'s with the supervision weakened: the same
1,000 training instances, but only their `min` outputs -- no oracle
labelling at box boundaries. The trained predicate is evaluated in the
exact fold with hard routing, next to its per-visit agreement with the
reference comparator it was never shown.

Run with `python -m goi.run_endtoend`.
"""

import time

import numpy as np

from goi import adapter
from goi import endtoend
from goi import minimum
from goi.run_minimum import SPLITS, per_visit_accuracy, score


def experiment(seed):
  """One seed: train on outputs alone, score with hard routing."""
  start = time.time()
  feedback = adapter.sample('minimum', 16, 1000, seed)
  keys = adapter.input_data(feedback, 'key')
  labels = np.argmax([
      data_point.data for data_point in feedback.outputs
      if data_point.name == 'min'][0], axis=-1)
  params = endtoend.train(keys, labels, seed)
  box = minimum.predicate_box(params)
  accuracy = per_visit_accuracy(box, 'wide test (n=64)')
  print(f"seed {seed}: per-visit agreement with the reference comparator, "
        f"never shown: {accuracy:.6f}")
  scores = {}
  for split in SPLITS:
    scores[split] = score(box, split)
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
  main()

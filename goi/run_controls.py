"""Run the parameter and wiring controls for the `minimum` result.

Run with `python -m goi.run_controls`.
"""

import time

import numpy as np

from goi import adapter
from goi import controls
from goi import minimum
from goi.run_minimum import SPLITS


def scores(evaluate):
  return {split: evaluate(split) for split in SPLITS}


def main(seed=0):
  start = time.time()
  feedback = adapter.sample('minimum', 16, 1000, seed)
  keys = adapter.input_data(feedback, 'key')
  oracle = minimum.RecordingOracle(minimum.reference)
  minimum.run(oracle, keys)

  small = minimum.predicate_box(minimum.train_predicate(oracle.visits, seed))
  sizes = controls.matched_sizes()
  inputs, targets = minimum.training_set(oracle.visits)
  big = minimum.predicate_box(minimum.train(
      minimum.cross_entropy, sizes, inputs, targets, seed,
      steps=5000, batch_size=256, tail=2000))
  budget = sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))
  print(f"parameter-matched predicate trained: {budget:,} parameters "
        f"({time.time() - start:.0f}s)")

  def on_comb(box):
    def evaluate(split):
      length, batch_size, sampler_seed = SPLITS[split]
      fb = adapter.sample('minimum', length, batch_size, sampler_seed)
      ks = adapter.input_data(fb, 'key')
      _, pos = minimum.run(box, ks)
      return adapter.score_mask_one(fb, 'min', pos)
    return evaluate

  def on_tree(split):
    length, batch_size, sampler_seed = SPLITS[split]
    fb = adapter.sample('minimum', length, batch_size, sampler_seed)
    ks = adapter.input_data(fb, 'key')
    _, pos = controls.run_tree(small, ks)
    return adapter.score_mask_one(fb, 'min', pos)

  def on_half(split):
    length, batch_size, sampler_seed = SPLITS[split]
    fb = adapter.sample('minimum', length, batch_size, sampler_seed)
    ks = adapter.input_data(fb, 'key')
    _, pos = minimum.run(small, ks[:, :length // 2])
    return adapter.score_mask_one(fb, 'min', pos)

  results = {
      '322-parameter predicate, left comb': scores(on_comb(small)),
      f'{budget:,}-parameter predicate, left comb': scores(on_comb(big)),
      '322-parameter predicate, balanced tournament': scores(on_tree),
      '322-parameter predicate, half the keys folded': scores(on_half)}
  for name, result in results.items():
    print(name)
    for split, value in result.items():
      print(f"  {split}: {value:.4f}")

  print(f"smallest instantiable Triplet-GMPNN: "
        f"{controls.smallest_baseline():,} parameters")
  print(f"({time.time() - start:.0f}s)")


if __name__ == '__main__':
  main()

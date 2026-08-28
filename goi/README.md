# Map neural networks on CLRS

Stage 1 of [discopy#678](https://github.com/discopy/discopy/issues/678):
the beachhead adapter and one task end to end, porting the
geometry-of-interaction learning pipeline of
[discopy#677](https://github.com/discopy/discopy/pull/677) from Church
arithmetic to the [CLRS Algorithmic Reasoning
Benchmark](https://github.com/google-deepmind/clrs).

The thesis on CLRS's turf: a fully-connected GNN processor does not
enforce the algorithm's dataflow — in a map neural network **the wiring
is the algorithm**, and only the primitive boxes are learned, supervised
at their own boundaries from the reference implementation's traffic.
CLRS's test axis — train at n = 16, test out of distribution at n = 64 —
then probes structure the model has by construction: the boxes never
see n.

## The `minimum` task

CLRS's reference `minimum` is a left fold of one comparison. Here it is
the left-comb diagram over one generator `min2 : pair @ pair -> pair` in
the free symmetric monoidal category (a pair is a key wire and a
position wire), run by a [DisCoPy](https://github.com/discopy/discopy)
functor into Python functions with JAX arrays on the wires.

The first thing #678 asked to establish — comparator keys are
real-valued, so the rule tables must quotient over key values — holds on
the recorded traffic: every visit of the reference box copies one of its
two input pairs exactly, and which one is a function of the predicate
`key1 > key2` alone. Fifteen thousand visits collapse to **two rules**;
the one learned component is a two-layer MLP for the predicate (Adam
with a Polyak-averaged tail), with the routing kept exact.

Scored by `clrs._src.evaluation.evaluate`, the exact function behind the
published numbers, over model seeds 0, 1, 2:

| split | predicate box | value-bottleneck ablation |
|---|---|---|
| val, 32 × n=16 | 100.00 ± 0.00 | 98.96 ± 1.47 |
| test, 32 × n=64 (CLRS protocol) | **95.83 ± 1.47** | 85.42 ± 8.96 |
| wide test, 1000 × n=64 | 98.67 ± 1.05 | 83.30 ± 9.61 |
| far, 32 × n=256 | **97.92 ± 1.47** | 40.62 ± 33.17 |

The published baseline on this task and protocol is Triplet-GMPNN at
96.08 ± 2.07 ([Ibarz et al. 2022](https://arxiv.org/abs/2209.11142),
table 2) — statistical par at n = 64, with essentially no further
degradation at n = 256, sixteen times the training size. The learned
predicate agrees with the reference on 99.91–99.97% of the n = 64
traffic and all its errors lie in a band |key1 − key2| < 0.005; the
residual score gap *is* that band, since the routing is exact.

The ablation is the same wiring with the box regressing the smaller key
instead of routing — the pointer must be decoded back from the predicted
value, and accumulated value error collapses out of distribution. The
discrete interface is what generalizes.

Honest mismatches, as recorded in #678: this scores CLRS's output
metric only — per-box visits are supervision at box boundaries, not
CLRS's hint trajectories — and `minimum` is the multiplicative instance
of the map-neural-network statement, not the lambda-calculus machine
of #677, which stays the theory exhibit.

## Run it

Sampling goes through `clrs._src.samplers` on the fly, so the heavy
dataset dependencies (TensorFlow, tfds) are not needed — `goi.stubs`
registers import-time placeholders for them.

```shell
python -m venv .venv && . .venv/bin/activate
pip install numpy jax attrs absl-py chex dm-haiku optax ml_collections \
    six opt-einsum pytest
pip install git+https://github.com/discopy/discopy@main
python -m goi.run_minimum   # ~75s per seed on CPU
python -m pytest goi/minimum_test.py
```

## Next, per the staged plan

`insertion_sort`/`bubble_sort` (comparator maps), `lcs_length` or
`matrix_chain_order` (message passing on a grid map), then
`binary_search` as the feedback/stream instance, where data-dependent
routing is the token machine's native mode.

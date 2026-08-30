"""Map neural networks on the CLRS Algorithmic Reasoning Benchmark.

The experiment of https://github.com/discopy/discopy/issues/678: fix a
string diagram whose wiring *is* the algorithm, choose JAX arrays as wire
semantics, run the diagram by functorial evaluation and learn only the
primitive boxes, supervised at box boundaries from the reference
implementation's own traffic (geometry-of-interaction compositionality).
Out-of-distribution size generalization -- CLRS's test axis, train at
n = 16 and test at n = 64 -- is then structural: the boxes never see n.
"""

# TODO

> keep on going with the neural GoI experiments, previous session left a handoff for you

The hand-off: [memory#66 comment](https://github.com/toumix/memory/pull/66#issuecomment-5450774641).
The plan being executed here, stage 1: [discopy#678](https://github.com/discopy/discopy/issues/678).

- [x] CI on [discopy#677](https://github.com/discopy/discopy/pull/677) and
  [discopy#401](https://github.com/discopy/discopy/pull/401) checked green, both re-subscribed,
  #677 marked ready for review
- [x] An adapter pulling data and reference traces from `clrs._src.samplers`,
  without the heavy dataset dependencies
- [x] One task end to end as a map neural network: the wiring is the algorithm,
  only the primitive boxes are learned
- [x] Oracle labelling at box boundaries, rule tables quotiented over key values,
  per-box MLPs trained in CPU JAX
- [x] Train at n=16, test at n=64, scored with CLRS's own metric
  next to the published baselines
- [ ] Draft PR on `toumix/clrs`, hand-off updated on the memory day PR

"""Unit tests for `goi.minimum`, sized to run in seconds on CPU."""

import numpy as np

from absl.testing import absltest

from goi import adapter
from goi import minimum


class MinimumTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.feedback = adapter.sample('minimum', 8, 200, seed=0)
    self.keys = adapter.input_data(self.feedback, 'key')
    self.oracle = minimum.RecordingOracle(minimum.reference)
    _, self.pos = minimum.run(self.oracle, self.keys)

  def test_reference_fold_is_exact(self):
    np.testing.assert_array_equal(self.pos, np.argmin(self.keys, axis=1))
    self.assertEqual(
        adapter.score_mask_one(self.feedback, 'min', self.pos), 1.0)

  def test_rule_table_has_two_rules(self):
    self.assertLen(minimum.rule_table(self.oracle.visits), 2)

  def test_learned_predicate_generalizes(self):
    params = minimum.train_predicate(
        self.oracle.visits, seed=0, steps=1500, tail=500)
    box = minimum.predicate_box(params)
    feedback = adapter.sample('minimum', 16, 64, seed=1)
    _, pos = minimum.run(box, adapter.input_data(feedback, 'key'))
    self.assertGreaterEqual(
        adapter.score_mask_one(feedback, 'min', pos), 0.9)

  def test_end_to_end_discovers_the_comparator(self):
    from goi import endtoend
    labels = np.argmax(self.feedback.outputs[0].data, axis=-1)
    params = endtoend.train(
        self.keys, labels, seed=0, steps=2500, tail=1000)
    box = minimum.predicate_box(params)
    feedback = adapter.sample('minimum', 16, 64, seed=1)
    _, pos = minimum.run(box, adapter.input_data(feedback, 'key'))
    self.assertGreaterEqual(
        adapter.score_mask_one(feedback, 'min', pos), 0.9)

  def test_bottleneck_decodes_from_values(self):
    params = minimum.train_bottleneck(
        self.oracle.visits, seed=0, steps=1500, tail=500)
    box = minimum.bottleneck_box(params)
    feedback = adapter.sample('minimum', 8, 64, seed=1)
    keys = adapter.input_data(feedback, 'key')
    value, _ = minimum.run(box, keys)
    pos = minimum.decode_nearest(keys, value)
    self.assertGreaterEqual(
        adapter.score_mask_one(feedback, 'min', pos), 0.7)


if __name__ == '__main__':
  absltest.main()

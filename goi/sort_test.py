"""Unit tests for `goi.sort`, sized to run in seconds on CPU."""

import numpy as np

from absl.testing import absltest

from goi import adapter
from goi import sort


class SortTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.feedback = adapter.sample('insertion_sort', 8, 200, seed=0)
    self.keys = adapter.input_data(self.feedback, 'key')
    self.oracle = sort.RecordingOracle(sort.reference)
    self.sorted_pos = sort.run(self.oracle, self.keys)

  def test_reference_network_is_exact(self):
    np.testing.assert_array_equal(
        self.sorted_pos, np.argsort(self.keys, axis=1))
    self.assertEqual(adapter.score_pointer(
        self.feedback, 'pred', sort.predecessors(self.sorted_pos)), 1.0)

  def test_rule_table_has_two_rules(self):
    self.assertLen(sort.rule_table(self.oracle.visits), 2)

  def test_learned_predicate_generalizes(self):
    params = sort.train_predicate(
        self.oracle.visits, seed=0, steps=3000, tail=1000)
    box = sort.predicate_box(params)
    feedback = adapter.sample('insertion_sort', 16, 32, seed=1)
    keys = adapter.input_data(feedback, 'key')
    pointers = sort.predecessors(sort.run(box, keys))
    self.assertGreaterEqual(
        adapter.score_pointer(feedback, 'pred', pointers), 0.7)


if __name__ == '__main__':
  absltest.main()

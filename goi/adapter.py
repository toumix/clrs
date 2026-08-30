"""Sample CLRS tasks on the fly and score predictions with CLRS's own metric.

The adapter goes through `clrs._src.samplers` only, so no pre-generated
dataset and no TensorFlow are needed; scoring goes through
`clrs._src.evaluation.evaluate`, the exact function behind the published
benchmark numbers.
"""

import numpy as np

from goi import stubs

stubs.install()

# pylint: disable=g-import-not-at-top,wrong-import-position
from clrs._src import evaluation
from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs


def sample(algorithm, length, batch_size, seed):
  """One batch of `algorithm` instances of size `length`, as a Feedback."""
  sampler, _ = samplers.build_sampler(
      algorithm, num_samples=-1, length=length, seed=seed)
  return sampler.next(batch_size)


def input_data(feedback, name):
  """The data of the named input probe, shape [batch, length]."""
  for data_point in feedback.features.inputs:
    if data_point.name == name:
      return data_point.data
  raise KeyError(name)


def mask_one(indices, length):
  """One-hot rows for a batch of indices, the shape CLRS scores mask_one."""
  return np.eye(length, dtype=np.float32)[np.asarray(indices, dtype=int)]


def score_mask_one(feedback, name, indices):
  """CLRS's own score for a mask_one output predicted as argmax indices."""
  length = feedback.outputs[0].data.shape[-1]
  prediction = probing.DataPoint(
      name=name, location=specs.Location.NODE, type_=specs.Type.MASK_ONE,
      data=mask_one(indices, length))
  return evaluation.evaluate(feedback.outputs, {name: prediction})['score']


def score_pointer(feedback, name, pointers):
  """CLRS's own score for a pointer output predicted as indices.

  A `should_be_permutation` truth is relabelled to plain pointers, the
  `enforce_permutations=False` branch of `samplers.process_permutations`.
  """
  outputs = tuple(
      probing.DataPoint(
          name=data_point.name, location=data_point.location,
          type_=specs.Type.POINTER, data=data_point.data)
      if data_point.type_ == specs.Type.SHOULD_BE_PERMUTATION
      else data_point for data_point in feedback.outputs)
  truth_dtype = feedback.outputs[0].data.dtype
  prediction = probing.DataPoint(
      name=name, location=specs.Location.NODE, type_=specs.Type.POINTER,
      data=np.asarray(pointers, dtype=truth_dtype))
  return evaluation.evaluate(outputs, {name: prediction})['score']


def score_categorical(feedback, name, data):
  """CLRS's own score for an edge-categorical output, given as one-hot."""
  prediction = probing.DataPoint(
      name=name, location=specs.Location.EDGE, type_=specs.Type.CATEGORICAL,
      data=data)
  return evaluation.evaluate(feedback.outputs, {name: prediction})['score']

"""Import-time placeholders so `clrs` loads without TensorFlow.

Only the pre-generated dataset path (`clrs._src.dataset`) needs TensorFlow
and `tensorflow_datasets`; sampling on the fly with `clrs._src.samplers`
does not. `install()` registers minimal placeholder modules so that
`import clrs` succeeds in a CPU-light environment; it is a no-op whenever
the real packages are importable.
"""

import dataclasses
import importlib.util
import sys
import types


class _Anything:
  """An object whose every attribute, call and subclass check is itself."""

  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return self

  def __getattr__(self, name):
    return self


class _StubModule(types.ModuleType):

  def __getattr__(self, name):
    return _Anything()


@dataclasses.dataclass
class _BuilderConfig:
  """Base accepting the `name` that `dataset.CLRSConfig` is built with."""
  name: str = ''


def _missing(name):
  return name not in sys.modules and importlib.util.find_spec(name) is None


def install():
  """Register the placeholder modules for the packages that are missing."""
  if _missing('tensorflow'):
    tensorflow = _StubModule('tensorflow')
    tensorflow.Tensor = type('Tensor', (), {})
    tensorflow.is_tensor = lambda value: False
    sys.modules['tensorflow'] = tensorflow
  if _missing('tensorflow_datasets'):
    tfds = _StubModule('tensorflow_datasets')
    tfds.core = types.SimpleNamespace(
        BuilderConfig=_BuilderConfig,
        GeneratorBasedBuilder=object,
        Version=_Anything(),
        DatasetInfo=_Anything())
    sys.modules['tensorflow_datasets'] = tfds

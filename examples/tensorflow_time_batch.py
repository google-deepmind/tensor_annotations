# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example demonstrating checking of Time/Batch consistency in TensorFlow."""

from typing import cast

from absl import app
from tensor_annotations import axes
from tensor_annotations import tensorflow as ttf
import tensorflow as tf

# pylint: disable=missing-function-docstring

float16 = ttf.float16
Batch = axes.Batch
Time = axes.Time


def sample_batch() -> ttf.Tensor2[float16, Time, Batch]:
  # tf.zeros((x, y)) returns a Tensor2[Any, Any], which is a compatibe
  # with Tensor2[Batch, Time] => pytype accepts this return.
  return tf.zeros((3, 5))


# An example of legacy code annotated with a conventional tensor type rather
# than the shape-annotated version.
def sample_batch_legacy() -> tf.Tensor:
  # Even with our custom stubs, tf.zeros([...]) (with a list-shape!) returns an
  # unspecific `Any` type, so the type-checker is happy interpreting it as
  # tf.Tensor.
  return tf.zeros([3, 5])


def train_batch(batch: ttf.Tensor2[float16, Batch, Time]):
  b: ttf.Tensor1[float16, Batch] = tf.reduce_max(batch, axis=1)
  del b   # Unused


# `-> None` is necessary for Mypy to check this function
def transpose_example() -> None:
  # From the signature of sample_batch(), x1 is inferred to be of type
  # Tensor2[Time, Batch].
  x1 = sample_batch()

  # Using our custom stubs for tf.transpose(...), x2 is inferred to be of type
  # Tensor2[Batch, Time].
  x2 = tf.transpose(x1)

  # Tensor2[Batch, Time] is compatible with the signature of train_batch(),
  # so we're good! :)
  # Try changing `x2` to `x1` - you should find that this script no longer
  # passes type check.
  train_batch(x2)


# `-> None` is necessary for Mypy to check this function
def legacy_example() -> None:
  # From the signature of sample_batch_legacy(), y is inferred to be of
  # type tf.Tensor.
  y = sample_batch_legacy()

  # We explicitly cast it to the desired type. This is a no-op at runtime.
  y2 = cast(ttf.Tensor2[float16, Batch, Time], y)

  # Alternative syntax for casting; again a no-op.
  y3: ttf.Tensor2[float16, Batch, Time] = y  # type: ignore

  train_batch(y2)
  train_batch(y3)


def main(argv):
  del argv

  transpose_example()
  legacy_example()


if __name__ == '__main__':
  app.run(main)

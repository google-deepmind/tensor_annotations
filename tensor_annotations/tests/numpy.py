# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for NumPy stubs."""

from typing import cast, NewType

from absl.testing import absltest
import numpy as np
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tensor_annotations.numpy import AnyDType
from tensor_annotations.numpy import Array1
from tensor_annotations.numpy import int16
from tensor_annotations.numpy import int8
from tensor_annotations.tests import utils


A1 = NewType('A1', axes.Axis)

# It's less than ideal that we have to repeat imports etc. here for pytype, but
# this seems like the best balance between readability and complexity.
_PREAMBLE = """
from typing import cast, NewType

import numpy as np
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tensor_annotations.numpy import AnyDType, int8, int16
from tensor_annotations.numpy import Array1

A1 = NewType('A1', axes.Axis)
"""


class NumPyDtypeTests(absltest.TestCase):
  """Tests for data types inferred from NumPy type stubs using pytype."""

  def testFunctionWithInt8Argument_AcceptsInt8Value(self):
    """Tests whether a function will accept a value with the right dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1[int8, Batch]):
        pass
      x = cast(Array1[int8, Batch], np.array([0], dtype=np.int8))
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithInt8Argument_RejectsInt16Value(self):
    """Tests whether a function will reject a value with the wrong dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1[int8, Batch]):
        pass
      x = cast(Array1[int16, Batch], np.array([0], dtype=np.int16))
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)

  def testFunctionWithAnyDTypeArgument_AcceptsInt8Value(self):
    """Tests whether AnyDType makes a function argument compatible with all."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1[AnyDType, Batch]):
        pass
      x = cast(Array1[int8, Batch], np.array([0], dtype=np.int8))
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithInt8Argument_AcceptsAnyDTypeValue(self):
    """Tests whether AnyDType is compatible with an arbitrary argument dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1[int8, Batch]):
        pass
      x = cast(Array1[AnyDType, Batch], np.array([0]))
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)


if __name__ == '__main__':
  absltest.main()

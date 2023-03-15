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
from tensor_annotations.numpy import Array2
from tensor_annotations.numpy import int16
from tensor_annotations.numpy import int8
from tensor_annotations.tests import utils


A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)

# It's less than ideal that we have to repeat imports etc. here for pytype, but
# this seems like the best balance between readability and complexity.
_PREAMBLE = """
from typing import cast, NewType

import numpy as np
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tensor_annotations.numpy import AnyDType, int8, int16
from tensor_annotations.numpy import Array1, Array2

A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
"""


class NumPyStubTests(absltest.TestCase):
  """Tests for numpy.* stubs."""

  def testZerosOnes_ReturnsCorrectShape(self):
    """Confirms that np.zeros() returns a tensor_annotations type."""
    with utils.SaveCodeAsString() as code_saver:
      a = np.zeros((1,))  # pylint: disable=unused-variable
      b = np.ones((1,))  # pylint: disable=unused-variable
      c = np.zeros((1, 1))  # pylint: disable=unused-variable
      d = np.ones((1, 1))  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Array1')
    self.assertEqual(inferred.b, 'Array1')
    self.assertEqual(inferred.c, 'Array2')
    self.assertEqual(inferred.d, 'Array2')

  def testArrayUnaryOp_ReturnsCorrectTypeAndShape(self):
    """Confirms that unary functions like abs() don't change the shape."""
    with utils.SaveCodeAsString() as code_saver:
      x1 = cast(Array1[AnyDType, A1], np.array([0]))
      y1 = abs(x1)  # pylint: disable=unused-variable
      y2 = -x1  # pylint: disable=unused-variable
      x2 = cast(Array2[AnyDType, A1, A2], np.array([[0]]))
      y3 = abs(x2)  # pylint: disable=unused-variable
      y4 = -x2  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Array1[Any, A1]', inferred.y1)
    self.assertEqual('Array1[Any, A1]', inferred.y2)
    self.assertEqual('Array2[Any, A1, A2]', inferred.y3)
    self.assertEqual('Array2[Any, A1, A2]', inferred.y4)


class NumPyDtypeTests(absltest.TestCase):
  """Tests for data types inferred from NumPy type stubs using pytype."""

  def testZerosOnes_ReturnsAnyDType(self):
    """Tests that np.zeros and np.ones returns AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      a = np.zeros((1,))  # pylint: disable=unused-variable
      b = np.ones((1,))  # pylint: disable=unused-variable

      c = np.zeros((1, 1))  # pylint: disable=unused-variable
      d = np.ones((1, 1))  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # These should be e.g. Array1[AnyDType, Any], but because AnyDType
    # is currently aliased to `Any`, and pytype doesn't print type arguments at
    # all when they're all `Any`, hence just comparing to e.g. Array1.
    self.assertEqual(inferred.a, 'Array1')
    self.assertEqual(inferred.b, 'Array1')
    self.assertEqual(inferred.c, 'Array2')
    self.assertEqual(inferred.d, 'Array2')

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

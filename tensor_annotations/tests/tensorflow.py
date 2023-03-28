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
"""Tests for TensorFlow stubs."""

from typing import Any, NewType, SupportsFloat, TypeVar

from absl.testing import absltest  # For sharded test support
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tensor_annotations.axes import Time
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import AnyDType
from tensor_annotations.tensorflow import float32
from tensor_annotations.tensorflow import float64
from tensor_annotations.tensorflow import int16
from tensor_annotations.tensorflow import int8
from tensor_annotations.tensorflow import Tensor0
from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import Tensor1AnyDType
from tensor_annotations.tensorflow import Tensor2
from tensor_annotations.tests import utils
import tensorflow as tf


A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
A3 = NewType('A3', axes.Axis)
AxisTypeVar = TypeVar('AxisTypeVar')

# It's less than ideal that we have to repeat imports etc. here for pytype, but
# this seems like the best balance between readability and complexity.
_PREAMBLE = """
from typing import Any, NewType, SupportsFloat, TypeVar

import tensorflow as tf
from tensor_annotations import axes
from tensor_annotations.axes import Batch, Time
import tensor_annotations.tensorflow as ttf
from tensor_annotations.tensorflow import AnyDType
from tensor_annotations.tensorflow import float32, float64, int8, int16
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor1AnyDType, Tensor2

A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
A3 = NewType('A3', axes.Axis)
AxisTypeVar = TypeVar('AxisTypeVar')
"""


class TensorFlowShapeTests(absltest.TestCase):
  """Tests for shapes inferred from TensorFlow type stubs using pytype."""

  def testTranspose_InferredMatchesActualShapeShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      y = tf.transpose(x)

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, y.shape)

  def testUnaryOperator_ReturnCustomType(self):
    """Tests that operators like `tf.sin` return tensor_annotations types."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[Any, A1] = tf.zeros((1,))
      # Let's just test a representative subset.
      a = tf.abs(x)  # pylint: disable=unused-variable
      b = tf.sin(x)  # pylint: disable=unused-variable
      c = tf.floor(x)  # pylint: disable=unused-variable
      d = tf.ones_like(x)  # pylint: disable=unused-variable
      e = tf.round(x)  # pylint: disable=unused-variable
      f = tf.sign(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # Any is printed as Any in pytype output.
    expected = 'Tensor1[Any, A1]'
    self.assertEqual(inferred.a, expected)
    self.assertEqual(inferred.b, expected)
    self.assertEqual(inferred.c, expected)
    self.assertEqual(inferred.d, expected)
    self.assertEqual(inferred.e, expected)
    self.assertEqual(inferred.f, expected)

  def testMathUnaryOperator_ReturnCustomType(self):
    """Tests that operators like `tf.math.sin` return the correct types."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[Any, A1] = tf.zeros((1,))
      # Let's just test a representative subset.
      a = tf.math.abs(x)  # pylint: disable=unused-variable
      b = tf.math.sin(x)  # pylint: disable=unused-variable
      c = tf.math.floor(x)  # pylint: disable=unused-variable
      d = tf.math.round(x)  # pylint: disable=unused-variable
      e = tf.math.sign(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # Any is printed as Any in pytype output.
    expected = 'Tensor1[Any, A1]'
    self.assertEqual(inferred.a, expected)
    self.assertEqual(inferred.b, expected)
    self.assertEqual(inferred.c, expected)
    self.assertEqual(inferred.d, expected)
    self.assertEqual(inferred.e, expected)

  def testZerosOnes_ReturnsCorrectShape(self):
    """Tests that e.g. `tf.zeros` returns the correct types."""
    with utils.SaveCodeAsString() as code_saver:
      a = tf.zeros(())  # pylint: disable=unused-variable
      b = tf.ones(())  # pylint: disable=unused-variable
      c = tf.zeros((1,))  # pylint: disable=unused-variable
      d = tf.ones((1,))  # pylint: disable=unused-variable
      e = tf.zeros((1, 1))  # pylint: disable=unused-variable
      f = tf.ones((1, 1))  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Tensor0')
    self.assertEqual(inferred.b, 'Tensor0')
    self.assertEqual(inferred.c, 'Tensor1')
    self.assertEqual(inferred.d, 'Tensor1')
    self.assertEqual(inferred.e, 'Tensor2')
    self.assertEqual(inferred.f, 'Tensor2')

  def testSum_InferredMatchesActualShape(self):
    """Tests that `tf.reduce_sum` returns the correct types."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[Any, A1] = tf.zeros((1,))
      y: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      x0 = tf.reduce_sum(x, axis=0)
      y0 = tf.reduce_sum(y, axis=0)
      y1 = tf.reduce_sum(y, axis=1)
      yn1 = tf.reduce_sum(y, axis=-1)

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.x0, x0.shape)
    self.assertEqual(inferred.y0, y0.shape)
    self.assertEqual(inferred.y1, y1.shape)
    self.assertEqual(inferred.yn1, yn1.shape)

  def testMatmul_InferredMatchesActualShape(self):
    """Tests that `x @ y` returns the correct types."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      y: Tensor2[Any, A2, A3] = tf.zeros((2, 3))
      xy = x @ y

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.xy, xy.shape)

  def testTensorAdd_ReturnsCustomType(self):
    """Tests that addition returns the correct types."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[Any, A1] = tf.zeros((1,))
      a = x + 1  # pylint: disable=unused-variable
      b = x + x  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # Any is printed as Any in pytype output.
    self.assertEqual(inferred.a, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.b, 'Tensor1[Any, A1]')

  def testTensorUnaryOp_ReturnsCorrectTypeAndShape(self):
    """Tests that e.g. `-x` has the correct type."""
    with utils.SaveCodeAsString() as code_saver:
      x1: Tensor0[int16] = tf.zeros(())
      y1 = abs(x1)  # pylint: disable=unused-variable
      y2 = -x1  # pylint: disable=unused-variable
      x2: Tensor1[int16, A1] = tf.zeros((1,))
      y3 = abs(x2)  # pylint: disable=unused-variable
      y4 = -x2  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Tensor0[int16]', inferred.y1)
    self.assertEqual('Tensor0[int16]', inferred.y2)
    # Any is printed as Any in pytype output.
    self.assertEqual('Tensor1[int16, A1]', inferred.y3)
    self.assertEqual('Tensor1[int16, A1]', inferred.y4)

  def testBinaryOpWithScalar_InferredMatchesActualShape(self):
    """Tests that e.g. `x + 1` has the correct type."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      y1 = x + 1.0
      y2 = x - 1.0
      y3 = x / 1.0
      y4 = x * 1.0

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(y1.shape, inferred.y1)
    self.assertEqual(y2.shape, inferred.y2)
    self.assertEqual(y3.shape, inferred.y3)
    self.assertEqual(y4.shape, inferred.y4)

  def testBinaryOpWithBroadcast_InferredMatchesActualShape(self):
    """Tests the result of e.g. adding two tensors with different shapes."""
    with utils.SaveCodeAsString() as code_saver:
      a: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      b: Tensor1[Any, A2] = tf.zeros((2,))
      y1 = a + b
      y2 = a - b
      y3 = a / b
      y4 = a * b

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(y1.shape, inferred.y1)
    self.assertEqual(y2.shape, inferred.y2)
    self.assertEqual(y3.shape, inferred.y3)
    self.assertEqual(y4.shape, inferred.y4)

  def testBinaryOpWithSameShape_InferredMatchesActualShape(self):
    """Tests the result of e.g. adding two tensors with the same shape."""
    with utils.SaveCodeAsString() as code_saver:
      a: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      b: Tensor2[Any, A1, A2] = tf.zeros((1, 2))
      y1 = a + b
      y2 = a - b
      y3 = a / b
      y4 = a * b

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(y1.shape, inferred.y1)
    self.assertEqual(y2.shape, inferred.y2)
    self.assertEqual(y3.shape, inferred.y3)
    self.assertEqual(y4.shape, inferred.y4)

  def testShapeAttribute_HasTypeTensorShape(self):
    """Tests that `x.shape` is a tensorflow.TensorShape."""
    with utils.SaveCodeAsString() as code_saver:
      x0 = tf.zeros(())
      x1 = tf.zeros((1,))
      x2 = tf.zeros((1, 2))
      x3 = tf.zeros((1, 2, 3))
      x4 = tf.zeros((1, 2, 3, 4))
      x5 = tf.zeros((1, 2, 3, 4, 5))
      x0_shape = x0.shape   # pylint: disable=unused-variable
      x1_shape = x1.shape   # pylint: disable=unused-variable
      x2_shape = x2.shape   # pylint: disable=unused-variable
      x3_shape = x3.shape   # pylint: disable=unused-variable
      x4_shape = x4.shape   # pylint: disable=unused-variable
      x5_shape = x5.shape   # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('tensorflow.TensorShape', inferred.x0_shape)
    self.assertEqual('tensorflow.TensorShape', inferred.x1_shape)
    self.assertEqual('tensorflow.TensorShape', inferred.x2_shape)
    self.assertEqual('tensorflow.TensorShape', inferred.x3_shape)
    self.assertEqual('tensorflow.TensorShape', inferred.x4_shape)
    self.assertEqual('tensorflow.TensorShape', inferred.x5_shape)

  def testShapeAttribute_HasLen(self):
    with utils.SaveCodeAsString() as code_saver:
      x = tf.zeros((1,))
      rank = len(x.shape)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)
    self.assertEqual('int', inferred.rank)

  def testTensor0_CanBeConvertedToFloat(self):
    with utils.SaveCodeAsString() as code_saver:
      x = tf.zeros(())
      y = float(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, 'float')

  def testTensor0_SupportsFloat(self):
    with utils.SaveCodeAsString() as code_saver:

      def foo(x: SupportsFloat):
        return x

      x = tf.zeros(())
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)


class TensorFlowDtypeTests(absltest.TestCase):
  """Tests for data types inferred from TensorFlow type stubs using pytype."""

  def testTranspose_ReturnsSameDtypeAsInput(self):
    """Tests that tf.transpose() doesn't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x8: Tensor2[int8, A1, A1] = tf.constant([[0]], dtype=tf.int8)
      x16: Tensor2[int16, A1, A1] = tf.constant([[0]], dtype=tf.int16)
      y8 = tf.transpose(x8)  # pylint: disable=unused-variable
      y16 = tf.transpose(x16)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y8, 'Tensor2[int8, A1, A1]')
    self.assertEqual(inferred.y16, 'Tensor2[int16, A1, A1]')

  def testUnaryFunctions_ReturnSameDtypeAsInput(self):
    """Tests that functions like `tf.sin` don't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x32: Tensor1[float32, A1] = tf.constant([0.0], dtype=tf.float32)
      x64: Tensor1[float64, A1] = tf.constant([0.0], dtype=tf.float64)
      # Let's just test a representative subset.
      a32 = tf.abs(x32)  # pylint: disable=unused-variable
      a64 = tf.abs(x64)  # pylint: disable=unused-variable
      b32 = tf.sin(x32)  # pylint: disable=unused-variable
      b64 = tf.sin(x64)  # pylint: disable=unused-variable
      c32 = tf.floor(x32)  # pylint: disable=unused-variable
      c64 = tf.floor(x64)  # pylint: disable=unused-variable
      d32 = tf.round(x32)  # pylint: disable=unused-variable
      d64 = tf.round(x64)  # pylint: disable=unused-variable
      e32 = tf.sign(x32)  # pylint: disable=unused-variable
      e64 = tf.sign(x64)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.a64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.b32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.b64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.c32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.c64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.d32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.d64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.e32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.e64, 'Tensor1[float64, A1]')

  def testMathUnaryFunctions_ReturnSameDtypeAsInput(self):
    """Tests that functions like `tf.math.sin` don't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x32: Tensor1[float32, A1] = tf.constant([0.0], dtype=tf.float32)
      x64: Tensor1[float64, A1] = tf.constant([0.0], dtype=tf.float64)
      # Let's just test a representative subset.
      a32 = tf.math.abs(x32)  # pylint: disable=unused-variable
      a64 = tf.math.abs(x64)  # pylint: disable=unused-variable
      b32 = tf.math.sin(x32)  # pylint: disable=unused-variable
      b64 = tf.math.sin(x64)  # pylint: disable=unused-variable
      c32 = tf.math.floor(x32)  # pylint: disable=unused-variable
      c64 = tf.math.floor(x64)  # pylint: disable=unused-variable
      d32 = tf.math.round(x32)  # pylint: disable=unused-variable
      d64 = tf.math.round(x64)  # pylint: disable=unused-variable
      e32 = tf.math.sign(x32)  # pylint: disable=unused-variable
      e64 = tf.math.sign(x64)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.a64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.b32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.b64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.c32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.c64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.d32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.d64, 'Tensor1[float64, A1]')
    self.assertEqual(inferred.e32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.e64, 'Tensor1[float64, A1]')

  def testZerosOnes_ReturnsAnyDType(self):
    """Tests that tf.zeros and tf.ones returns AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      a = tf.zeros(())  # pylint: disable=unused-variable
      b = tf.ones(())  # pylint: disable=unused-variable
      ref0: Tensor0[AnyDType] = tf.constant(0)  # pylint: disable=unused-variable

      c = tf.zeros((1,))  # pylint: disable=unused-variable
      d = tf.ones((1,))  # pylint: disable=unused-variable
      ref1: Tensor1[AnyDType, Any] = tf.constant([0])  # pylint: disable=unused-variable

      e = tf.zeros((1, 1))  # pylint: disable=unused-variable
      f = tf.ones((1, 1))  # pylint: disable=unused-variable
      ref2: Tensor2[AnyDType, Any, Any] = tf.constant([[0]])  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # We can't compare explicitly to e.g. Tensor0[AnyDType], because AnyDType
    # is currently aliased to Any, and if all the type arguments are Any,
    # pytype doesn't print the type arguments at all.
    self.assertEqual(inferred.a, inferred.ref0)
    self.assertEqual(inferred.b, inferred.ref0)
    self.assertEqual(inferred.c, inferred.ref1)
    self.assertEqual(inferred.d, inferred.ref1)
    self.assertEqual(inferred.e, inferred.ref2)
    self.assertEqual(inferred.f, inferred.ref2)

  def testSum_ReturnsSameDtypeAsInput(self):
    """Tests that tf.reduce_sum() doesn't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x32: Tensor1[float32, A1] = tf.constant([0.0], dtype=tf.float32)  # pylint: disable=unused-variable
      x64: Tensor1[float64, A1] = tf.constant([0.0], dtype=tf.float64)  # pylint: disable=unused-variable
      y32: Tensor2[float32, A1, A1] = tf.constant([[0.0]], dtype=tf.float32)  # pylint: disable=unused-variable
      y64: Tensor2[float64, A1, A1] = tf.constant([[0.0]], dtype=tf.float64)  # pylint: disable=unused-variable
      xsum32 = tf.reduce_sum(x32, axis=0)  # pylint: disable=unused-variable
      xsum64 = tf.reduce_sum(x64, axis=0)  # pylint: disable=unused-variable
      ysum32 = tf.reduce_sum(y32, axis=0)  # pylint: disable=unused-variable
      ysum64 = tf.reduce_sum(y64, axis=0)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.xsum32, 'Tensor0[float32]')
    self.assertEqual(inferred.xsum64, 'Tensor0[float64]')
    self.assertEqual(inferred.ysum32, 'Tensor1[float32, A1]')
    self.assertEqual(inferred.ysum64, 'Tensor1[float64, A1]')

  def testTensorAdd_ReturnsAnyDType(self):
    """Tests that e.g. `x + 1` has dtype AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[int8, A1] = tf.constant([[0]])
      a = x + 1  # pylint: disable=unused-variable
      b = x + x  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # AnyDType is printed as Any in pytype output.
    self.assertEqual(inferred.a, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.b, 'Tensor1[Any, A1]')

  def testTensorUnaryOp_ReturnsSameDTypeAsInput(self):
    """Tests that e.g. `-x` has the same dtype as `x`."""
    with utils.SaveCodeAsString() as code_saver:
      a8: Tensor0[int8] = tf.constant([[0]], dtype=tf.int8)
      b8 = abs(a8)  # pylint: disable=unused-variable
      c8 = -a8  # pylint: disable=unused-variable

      a16: Tensor0[int16] = tf.constant([[0]], dtype=tf.int16)
      b16 = abs(a16)  # pylint: disable=unused-variable
      c16 = -a16  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b8, 'Tensor0[int8]')
    self.assertEqual(inferred.c8, 'Tensor0[int8]')
    self.assertEqual(inferred.b16, 'Tensor0[int16]')
    self.assertEqual(inferred.c16, 'Tensor0[int16]')

  def testBinaryOpWithScalar_ReturnsAnyDType(self):
    """Tests that e.g. `x + 1` has dtype AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      y1 = x + 1  # pylint: disable=unused-variable
      y2 = x - 1  # pylint: disable=unused-variable
      y3 = x / 1  # pylint: disable=unused-variable
      y4 = x * 1  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # pytype displays AnyDType as Any.
    self.assertEqual(inferred.y1, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.y2, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.y3, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.y4, 'Tensor1[Any, A1]')

  def testBinaryOpWithArray_ReturnsAnyDType(self):
    """Tests that e.g. adding two arrays results in dtype AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      a: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      b: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      y1 = a + b  # pylint: disable=unused-variable
      y2 = a - b  # pylint: disable=unused-variable
      y3 = a / b  # pylint: disable=unused-variable
      y4 = a * b  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # pytype displays AnyDType as Any.
    self.assertEqual(inferred.y1, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.y2, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.y3, 'Tensor1[Any, A1]')
    self.assertEqual(inferred.y4, 'Tensor1[Any, A1]')

  def testFunctionWithInt8Argument_AcceptsInt8Value(self):
    """Tests whether a function will accept a value with the right dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor0[int8]):
        pass
      x: Tensor0[int8] = tf.constant([0], dtype=tf.int8)
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithInt8Argument_RejectsInt16Value(self):
    """Tests whether a function will reject a value with the wrong dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor0[int8]):
        pass
      x: Tensor0[int16] = tf.constant([0], dtype=tf.int16)
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)

  def testFunctionWithAnyDTypeArgument_AcceptsInt8Value(self):
    """Tests whether AnyDType makes a function argument compatible with all."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor0[AnyDType]):
        pass
      x: Tensor0[int8] = tf.constant([0], dtype=tf.int8)
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithInt8Argument_AcceptsAnyDTypeValue(self):
    """Tests whether AnyDType is compatible with an arbitrary argument dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor0[int8]):
        pass
      x: Tensor0[AnyDType] = tf.constant([0])
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithBoolArgument_AcceptsBoolValue(self):
    """No problems with using 'bool' as a dtype name, right?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor0[ttf.bool]):
        pass
      x: Tensor0[ttf.bool] = tf.constant(False)
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testTensorShapeAttr_IsTensorShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      s = x.shape  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.s, 'tensorflow.TensorShape')

  def testTensorShapeIndexedWithInt_IsInt(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      s = x.shape
      s0 = s[0]  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.s0, 'int')

  def testTensorShapeIndexedWithSlice_IsTensorShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      s = x.shape
      s0 = s[:1]  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.s0, 'tensorflow.TensorShape')

  def testTensorShapeAsList_IsListOfInt(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[int8, A1] = tf.constant([0], dtype=tf.int8)
      s = x.shape
      l = s.as_list()  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.l, 'List[int]')


class TensorFlowAnyDtypeAliasTests(absltest.TestCase):
  """Tests for backwards-compatible aliases that don't use DTypes."""

  def testInt8Batch_AcceptsAnyDTypeBatch(self):
    """Is Tensor1AnyDType[Batch] compatible with Tensor1[int8, Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor1[int8, Batch]):
        pass
      x: Tensor1AnyDType[Batch] = tf.constant([[0]])
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testInt8Batch_RejectsAnyDTypeTime(self):
    """Is Tensor1AnyDType[Time] compatible with Tensor1[int8, Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor1[int8, Batch]):
        pass
      x: Tensor1AnyDType[Time] = tf.constant([[0]])
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)

  def testAnyDTypeBatch_AcceptsUint8Batch(self):
    """Is Tensor1[int8, Batch] compatible with Tensor1AnyDType[Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor1AnyDType[Batch]):
        pass
      x: Tensor1[int8, Batch] = tf.constant([[0]])
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testAnyDTypeBatch_RejectsUint8Time(self):
    """Is Tensor1[int8, Time] compatible with Tensor1AnyDType[Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Tensor1AnyDType[Batch]):
        pass
      x: Tensor1[int8, Time] = tf.constant([[0]])
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)


if __name__ == '__main__':
  absltest.main()

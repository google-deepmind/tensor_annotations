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
"""Tests for JAX stubs."""

from typing import cast, NewType, SupportsFloat, TypeVar

from absl.testing import absltest
import jax
import jax.numpy as jnp
from tensor_annotations import axes
from tensor_annotations.axes import Batch
from tensor_annotations.axes import Time
from tensor_annotations.jax import AnyDType
from tensor_annotations.jax import Array0
from tensor_annotations.jax import Array1
from tensor_annotations.jax import Array1AnyDType
from tensor_annotations.jax import Array2
from tensor_annotations.jax import float32
from tensor_annotations.jax import float64
from tensor_annotations.jax import int16
from tensor_annotations.jax import int8
from tensor_annotations.tests import utils


A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
A3 = NewType('A3', axes.Axis)
AxisTypeVar = TypeVar('AxisTypeVar')

# It's less than ideal that we have to repeat imports etc. here for pytype, but
# this seems like the best balance between readability and complexity.
_PREAMBLE = """
from typing import Any, cast, NewType, SupportsFloat, TypeVar, Union

import jax
import jax.numpy as jnp
from tensor_annotations import axes
from tensor_annotations.axes import Batch, Time
from tensor_annotations.jax import AnyDType, float32, float64, int16, int8
from tensor_annotations.jax import Array0, Array1, Array1AnyDType, Array2

A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
A3 = NewType('A3', axes.Axis)
AxisTypeVar = TypeVar('AxisTypeVar')
"""


class JAXStubTests(absltest.TestCase):
  """Tests for jax.* stubs."""

  def test_custom_stubs_are_used_for_jax(self):
    """Tests whether eg a syntax error in jax.pyi prevents stubs being used."""
    # _sentinel is a member that exists with a specific type in our stubs but
    # not in the JAX library code itself (and would therefore normally be
    # seen as `Any` by pytype).
    code = _PREAMBLE + 's = jax._sentinel'

    inferred = utils.pytype_infer_types(code)

    self.assertEqual(inferred.s, 'int')


class JAXNumpyStubTests(absltest.TestCase):
  """Tests for jax.numpy.* stubs."""

  def testTranspose_InferredShapeMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
      y = jnp.transpose(x)
      y2 = x.T

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, y.shape)
    self.assertEqual(inferred.y2, y2.shape)

  def testUnaryOperator_ReturnCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[AnyDType, A1] = jnp.zeros((1,))
      # Let's just test a representative subset.
      a = jnp.abs(x)  # pylint: disable=unused-variable
      b = jnp.sin(x)  # pylint: disable=unused-variable
      c = jnp.floor(x)  # pylint: disable=unused-variable
      d = jnp.ones_like(x)  # pylint: disable=unused-variable
      e = jnp.round(x)  # pylint: disable=unused-variable
      f = jnp.sign(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    expected = 'Array1[Any, A1]'
    self.assertEqual(inferred.a, expected)
    self.assertEqual(inferred.b, expected)
    self.assertEqual(inferred.c, expected)
    self.assertEqual(inferred.d, expected)
    self.assertEqual(inferred.e, expected)
    self.assertEqual(inferred.f, expected)

  def testZerosOnes_ReturnsCorrectShape(self):
    with utils.SaveCodeAsString() as code_saver:
      a = jnp.zeros(())  # pylint: disable=unused-variable
      b = jnp.ones(())  # pylint: disable=unused-variable
      c = jnp.zeros((1,))  # pylint: disable=unused-variable
      d = jnp.ones((1,))  # pylint: disable=unused-variable
      e = jnp.zeros((1, 1))  # pylint: disable=unused-variable
      f = jnp.ones((1, 1))  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Array0[float32]')
    self.assertEqual(inferred.b, 'Array0[float32]')
    self.assertEqual(inferred.c, 'Array1[float32, Any]')
    self.assertEqual(inferred.d, 'Array1[float32, Any]')
    self.assertEqual(inferred.e, 'Array2[float32, Any, Any]')
    self.assertEqual(inferred.f, 'Array2[float32, Any, Any]')

  def testSum_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
      y1 = jnp.sum(x, axis=0)
      y2 = jnp.sum(x, axis=1)
      y3 = jnp.sum(x, axis=(0, 1))
      y4 = jnp.sum(x)

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y1, y1.shape)
    self.assertEqual(inferred.y2, y2.shape)
    self.assertEqual(inferred.y3, y3.shape)
    self.assertEqual(inferred.y4, y4.shape)

  def testSumKeepdimsTrue_ReturnsAny(self):
    # We haven't got around to making stubs for keepdims=True yet;
    # make sure the type reflects that.
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[AnyDType, A1] = jnp.zeros((1,))
      a = jnp.sum(x, axis=0, keepdims=True)  # pylint: disable=unused-variable
      b = jnp.sum(x, keepdims=True)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Any')
    self.assertEqual(inferred.b, 'Any')

  def testTensorAdd_ReturnsCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[AnyDType, A1] = jnp.zeros((1,))
      a = x + 1  # pylint: disable=unused-variable
      b = x + x  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Array1[Any, A1]', inferred.a)
    self.assertEqual('Array1[Any, A1]', inferred.b)

  def testMatmul_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
      y: Array2[AnyDType, A2, A3] = jnp.zeros((2, 3))
      xy = x @ y

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.xy, xy.shape)

  def testTensorUnaryOp_ReturnsCorrectTypeAndShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x1: Array0 = jnp.zeros(())
      y1 = abs(x1)  # pylint: disable=unused-variable
      y2 = -x1  # pylint: disable=unused-variable
      x2: Array1[AnyDType, A1] = jnp.zeros((1,))
      y3 = abs(x2)  # pylint: disable=unused-variable
      y4 = -x2  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Array0[DType]', inferred.y1)
    self.assertEqual('Array0[DType]', inferred.y2)
    self.assertEqual('Array1[Any, A1]', inferred.y3)
    self.assertEqual('Array1[Any, A1]', inferred.y4)

  def testBinaryOpWithScalar_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
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
    with utils.SaveCodeAsString() as code_saver:
      a: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
      b: Array1[AnyDType, A2] = jnp.zeros((2,))
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
    with utils.SaveCodeAsString() as code_saver:
      a: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
      b: Array2[AnyDType, A1, A2] = jnp.zeros((1, 2))
      y1 = a + b
      y2 = a - b
      y3 = a / b
      y4 = a * b

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(y1.shape, inferred.y1)
    self.assertEqual(y2.shape, inferred.y2)
    self.assertEqual(y3.shape, inferred.y3)
    self.assertEqual(y4.shape, inferred.y4)

  def testShapeAttribute_HasCorrectLength(self):
    with utils.SaveCodeAsString() as code_saver:
      x0 = jnp.zeros(())
      x1 = jnp.zeros((1,))
      x2 = jnp.zeros((1, 2))
      x3 = jnp.zeros((1, 2, 3))
      x4 = jnp.zeros((1, 2, 3, 4))
      x0_shape = x0.shape   # pylint: disable=unused-variable
      x1_shape = x1.shape   # pylint: disable=unused-variable
      x2_shape = x2.shape   # pylint: disable=unused-variable
      x3_shape = x3.shape   # pylint: disable=unused-variable
      x4_shape = x4.shape   # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(x0_shape, ())
    self.assertEqual(x1_shape, (1,))
    self.assertEqual(x2_shape, (1, 2))
    self.assertEqual(x3_shape, (1, 2, 3))
    self.assertEqual(x4_shape, (1, 2, 3, 4))
    self.assertEqual('Tuple[()]', inferred.x0_shape)
    self.assertEqual('Tuple[int]', inferred.x1_shape)
    self.assertEqual('Tuple[int, int]', inferred.x2_shape)
    self.assertEqual('Tuple[int, int, int]', inferred.x3_shape)
    self.assertEqual('Tuple[int, int, int, int]', inferred.x4_shape)

  def testArray0Item_ReturnsIntFloatBoolComplexUnion(self):
    with utils.SaveCodeAsString() as code_saver:
      x = jnp.zeros(())
      y = x.item()  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, 'Union[bool, complex, float, int]')

  def testArray0_CanBeConvertedToFloat(self):
    with utils.SaveCodeAsString() as code_saver:
      x = jnp.zeros(())
      y = float(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, 'float')

  def testArray0_SupportsFloat(self):
    with utils.SaveCodeAsString() as code_saver:

      def foo(x: SupportsFloat):
        return x

      x = jnp.zeros(())
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)


class JAXDtypeTests(absltest.TestCase):
  """Tests for data types inferred from JAX type stubs using pytype."""

  def testTranspose_ReturnsSameDtypeAsInput(self):
    """Tests that jnp.transpose() doesn't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x8: Array2[int8, A1, A1] = jnp.array([[0]], dtype=jnp.int8)
      x16: Array2[int16, A1, A1] = jnp.array([[0]], dtype=jnp.int16)
      y8 = jnp.transpose(x8)  # pylint: disable=unused-variable
      y16 = jnp.transpose(x16)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y8, 'Array2[int8, A1, A1]')
    self.assertEqual(inferred.y16, 'Array2[int16, A1, A1]')

  def testUnaryFunctions_ReturnSameDtypeAsInput(self):
    """Tests that functions like `jnp.sin` don't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x32: Array1[float32, A1] = jnp.array([0.0], dtype=jnp.float32)
      x64: Array1[float64, A1] = jnp.array([0.0], dtype=jnp.float64)
      # Let's just test a representative subset.
      a32 = jnp.abs(x32)  # pylint: disable=unused-variable
      a64 = jnp.abs(x64)  # pylint: disable=unused-variable
      b32 = jnp.sin(x32)  # pylint: disable=unused-variable
      b64 = jnp.sin(x64)  # pylint: disable=unused-variable
      c32 = jnp.floor(x32)  # pylint: disable=unused-variable
      c64 = jnp.floor(x64)  # pylint: disable=unused-variable
      d32 = jnp.round(x32)  # pylint: disable=unused-variable
      d64 = jnp.round(x64)  # pylint: disable=unused-variable
      e32 = jnp.sign(x32)  # pylint: disable=unused-variable
      e64 = jnp.sign(x64)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a32, 'Array1[float32, A1]')
    self.assertEqual(inferred.a64, 'Array1[float64, A1]')
    self.assertEqual(inferred.b32, 'Array1[float32, A1]')
    self.assertEqual(inferred.b64, 'Array1[float64, A1]')
    self.assertEqual(inferred.c32, 'Array1[float32, A1]')
    self.assertEqual(inferred.c64, 'Array1[float64, A1]')
    self.assertEqual(inferred.d32, 'Array1[float32, A1]')
    self.assertEqual(inferred.d64, 'Array1[float64, A1]')
    self.assertEqual(inferred.e32, 'Array1[float32, A1]')
    self.assertEqual(inferred.e64, 'Array1[float64, A1]')

  def testZerosOnes_ReturnsCorrectDtype(self):
    """Tests that jnp.zeros and jnp.ones returns arrays with correct dtypes."""
    with utils.SaveCodeAsString() as code_saver:
      a = jnp.zeros(())  # pylint: disable=unused-variable
      b = jnp.ones(())  # pylint: disable=unused-variable
      c = jnp.zeros((), dtype=jnp.int8)  # pylint: disable=unused-variable
      d = jnp.ones((), dtype=jnp.int8)  # pylint: disable=unused-variable

      e = jnp.zeros((1,))  # pylint: disable=unused-variable
      f = jnp.ones((1,))  # pylint: disable=unused-variable
      g = jnp.zeros((1,), dtype=jnp.int8)  # pylint: disable=unused-variable
      h = jnp.ones((1,), dtype=jnp.int8)  # pylint: disable=unused-variable

      i = jnp.zeros((1, 1))  # pylint: disable=unused-variable
      j = jnp.ones((1, 1))  # pylint: disable=unused-variable
      k = jnp.zeros((1, 1), dtype=jnp.int8)  # pylint: disable=unused-variable
      l = jnp.ones((1, 1), dtype=jnp.int8)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Array0[float32]')
    self.assertEqual(inferred.b, 'Array0[float32]')
    # These should be Array0[AnyDType], but because AnyDType is currently
    # aliased to `Any`, and pytype doesn't print type arguments at all when
    # they're all `Any`, hence just comparing to e.g. Array0.
    # Ditto tests below.
    self.assertEqual(inferred.c, 'Array0')
    self.assertEqual(inferred.d, 'Array0')

    self.assertEqual(inferred.e, 'Array1[float32, Any]')
    self.assertEqual(inferred.f, 'Array1[float32, Any]')
    self.assertEqual(inferred.g, 'Array1')
    self.assertEqual(inferred.h, 'Array1')

    self.assertEqual(inferred.i, 'Array2[float32, Any, Any]')
    self.assertEqual(inferred.j, 'Array2[float32, Any, Any]')
    self.assertEqual(inferred.k, 'Array2')
    self.assertEqual(inferred.l, 'Array2')

  def testSum_ReturnsSameDtypeAsInput(self):
    """Tests that jnp.sum() doesn't change the dtype."""
    with utils.SaveCodeAsString() as code_saver:
      x32: Array1[float32, A1] = jnp.array([0.0], dtype=jnp.float32)  # pylint: disable=unused-variable
      x64: Array1[float64, A1] = jnp.array([0.0], dtype=jnp.float64)  # pylint: disable=unused-variable
      y32: Array2[float32, A1, A1] = jnp.array([[0.0]], dtype=jnp.float32)  # pylint: disable=unused-variable
      y64: Array2[float64, A1, A1] = jnp.array([[0.0]], dtype=jnp.float64)  # pylint: disable=unused-variable
      xsum32 = jnp.sum(x32, axis=0)  # pylint: disable=unused-variable
      xsum64 = jnp.sum(x64, axis=0)  # pylint: disable=unused-variable
      ysum32 = jnp.sum(y32, axis=0)  # pylint: disable=unused-variable
      ysum64 = jnp.sum(y64, axis=0)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.xsum32, 'Array0[float32]')
    self.assertEqual(inferred.xsum64, 'Array0[float64]')
    self.assertEqual(inferred.ysum32, 'Array1[float32, A1]')
    self.assertEqual(inferred.ysum64, 'Array1[float64, A1]')

  def testArrayAdd_ReturnsAnyDType(self):
    """Tests that e.g. `x + 1` has dtype AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[int8, A1] = jnp.array([[0]])
      a = x + 1  # pylint: disable=unused-variable
      b = x + x  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # AnyDType is printed as Any in pytype output.
    self.assertEqual(inferred.a, 'Array1[Any, A1]')
    self.assertEqual(inferred.b, 'Array1[Any, A1]')

  def testArrayUnaryOp_ReturnsSameDTypeAsInput(self):
    """Tests that e.g. `-x` has the same dtype as `x`."""
    with utils.SaveCodeAsString() as code_saver:
      a8: Array0[int8] = jnp.array([[0]], dtype=jnp.int8)
      b8 = abs(a8)  # pylint: disable=unused-variable
      c8 = -a8  # pylint: disable=unused-variable

      a16: Array0[int16] = jnp.array([[0]], dtype=jnp.int16)
      b16 = abs(a16)  # pylint: disable=unused-variable
      c16 = -a16  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b8, 'Array0[int8]')
    self.assertEqual(inferred.c8, 'Array0[int8]')
    self.assertEqual(inferred.b16, 'Array0[int16]')
    self.assertEqual(inferred.c16, 'Array0[int16]')

  def testBinaryOpWithScalar_ReturnsAnyDType(self):
    """Tests that e.g. `x + 1` has dtype AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[int8, A1] = jnp.array([0], dtype=jnp.int8)
      y1 = x + 1  # pylint: disable=unused-variable
      y2 = x - 1  # pylint: disable=unused-variable
      y3 = x / 1  # pylint: disable=unused-variable
      y4 = x * 1  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # pytype displays AnyDType as Any.
    self.assertEqual(inferred.y1, 'Array1[Any, A1]')
    self.assertEqual(inferred.y2, 'Array1[Any, A1]')
    self.assertEqual(inferred.y3, 'Array1[Any, A1]')
    self.assertEqual(inferred.y4, 'Array1[Any, A1]')

  def testBinaryOpWithArray_ReturnsAnyDType(self):
    """Tests that e.g. adding two arrays results in dtype AnyDType."""
    with utils.SaveCodeAsString() as code_saver:
      a: Array1[int8, A1] = jnp.array([0], dtype=jnp.int8)
      b: Array1[int8, A1] = jnp.array([0], dtype=jnp.int8)
      y1 = a + b  # pylint: disable=unused-variable
      y2 = a - b  # pylint: disable=unused-variable
      y3 = a / b  # pylint: disable=unused-variable
      y4 = a * b  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # pytype displays AnyDType as Any.
    self.assertEqual(inferred.y1, 'Array1[Any, A1]')
    self.assertEqual(inferred.y2, 'Array1[Any, A1]')
    self.assertEqual(inferred.y3, 'Array1[Any, A1]')
    self.assertEqual(inferred.y4, 'Array1[Any, A1]')

  def testFunctionWithInt8Argument_AcceptsInt8Value(self):
    """Tests whether a function will accept a value with the right dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array0[int8]):
        pass
      x: Array0[int8] = jnp.array([0], dtype=jnp.int8)
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithInt8Argument_RejectsInt16Value(self):
    """Tests whether a function will reject a value with the wrong dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array0[int8]):
        pass
      x: Array0[int16] = jnp.array([0], dtype=jnp.int16)
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)

  def testFunctionWithAnyDTypeArgument_AcceptsInt8Value(self):
    """Tests whether AnyDType makes a function argument compatible with all."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array0[AnyDType]):
        pass
      x: Array0[int8] = jnp.array([0], dtype=jnp.int8)
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testFunctionWithInt8Argument_AcceptsAnyDTypeValue(self):
    """Tests whether AnyDType is compatible with an arbitrary argument dtype."""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array0[int8]):
        pass
      x: Array0[AnyDType] = jnp.array([0])
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)


class JAXAnyDtypeAliasTests(absltest.TestCase):
  """Tests for backwards-compatible aliases that don't use DTypes."""

  def testInt8Batch_AcceptsAnyDTypeBatch(self):
    """Is Array1AnyDType[Batch] compatible with Array1[int8, Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1[int8, Batch]):
        pass
      x: Array1AnyDType[Batch] = jnp.array([[0]])
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testInt8Batch_RejectsAnyDTypeTime(self):
    """Is Array1AnyDType[Time] compatible with Array1[int8, Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1[int8, Batch]):
        pass
      x: Array1AnyDType[Time] = jnp.array([[0]])
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)

  def testAnyDTypeBatch_AcceptsUint8Batch(self):
    """Is Array1[int8, Batch] compatible with Array1AnyDType[Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1AnyDType[Batch]):
        pass
      x: Array1[int8, Batch] = jnp.array([[0]])
      foo(x)

    utils.assert_pytype_succeeds(_PREAMBLE + code_saver.code)

  def testAnyDTypeBatch_RejectsUint8Time(self):
    """Is Array1[int8, Time] compatible with Array1AnyDType[Batch]?"""
    with utils.SaveCodeAsString() as code_saver:
      def foo(_: Array1AnyDType[Batch]):
        pass
      x: Array1[int8, Time] = jnp.array([[0]])
      foo(x)

    utils.assert_pytype_fails(_PREAMBLE + code_saver.code)


class JAXArrayTests(absltest.TestCase):
  """Test for operations on official jax.Array class.

  We need to do some explicit casting in these tests because, since we're
  using our stubs, things like jnp.zeros returns an ArrayN, rather than
  a jax.Array, as we want.
  """

  def testArrayShape_HasInferredTypeTupleInt(self):
    """Tests that pytype infers tuple[int, ...] for jax.Array.shape."""
    with utils.SaveCodeAsString() as code_saver:
      x = cast(jax.Array, jnp.zeros(3))
      s = x.shape  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.s, 'Tuple[int, ...]')

  def testArrayFunctions_ReturnJaxArray(self):
    """Tests that the inferred types for eg jax.Array.astype() is are right."""
    with utils.SaveCodeAsString() as code_saver:
      a = cast(jax.Array, jnp.zeros(3))
      b = a.astype(jnp.int64)
      c = a + 1
      d = 1 + a
      e = a - 1
      f = 1 - a
      g = a * 2
      h = 2 * a
      i = a / 2
      j = a // 2
      k = a ** 2
      l = a @ a
      m = a[0]
      n = a.T
      o = a.at[0].set(1)

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    # These are all jax.Arrays, right?
    # Or close enough - they'll actually be some subclass of jax.Array
    # like tensorflow.compiler.xla.python.xla_extension.Array.
    self.assertIsInstance(a, jax.Array)
    self.assertIsInstance(b, jax.Array)
    self.assertIsInstance(c, jax.Array)
    self.assertIsInstance(d, jax.Array)
    self.assertIsInstance(e, jax.Array)
    self.assertIsInstance(f, jax.Array)
    self.assertIsInstance(g, jax.Array)
    self.assertIsInstance(h, jax.Array)
    self.assertIsInstance(i, jax.Array)
    self.assertIsInstance(j, jax.Array)
    self.assertIsInstance(k, jax.Array)
    self.assertIsInstance(l, jax.Array)
    self.assertIsInstance(m, jax.Array)
    self.assertIsInstance(n, jax.Array)
    self.assertIsInstance(o, jax.Array)

    # If all the variables are definitely jax.Arrays, then we should have
    # inferred jax.Array types.
    self.assertEqual(inferred.a, 'jax.Array')
    self.assertEqual(inferred.b, 'jax.Array')
    self.assertEqual(inferred.c, 'jax.Array')
    self.assertEqual(inferred.d, 'jax.Array')
    self.assertEqual(inferred.e, 'jax.Array')
    self.assertEqual(inferred.f, 'jax.Array')
    self.assertEqual(inferred.g, 'jax.Array')
    self.assertEqual(inferred.h, 'jax.Array')
    self.assertEqual(inferred.i, 'jax.Array')
    self.assertEqual(inferred.j, 'jax.Array')
    self.assertEqual(inferred.k, 'jax.Array')
    self.assertEqual(inferred.l, 'jax.Array')
    self.assertEqual(inferred.m, 'jax.Array')
    self.assertEqual(inferred.n, 'jax.Array')
    self.assertEqual(inferred.o, 'jax.Array')

  def testUnaryFunction_ReturnsJaxArray(self):
    with utils.SaveCodeAsString() as code_saver:
      x = cast(jax.Array, jnp.zeros(3))
      y = jnp.abs(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, 'jax.Array')

  def testZerosLike_ReturnsJaxArray(self):
    """Tests that jnp.zeros_like(jax.Array) returns a jax.Array."""
    with utils.SaveCodeAsString() as code_saver:
      a = cast(jax.Array, jnp.zeros(3))
      # pylint: disable=unused-variable
      b = jnp.zeros_like(a)
      c = jnp.zeros_like(a, dtype=jnp.uint8)
      # pylint: enable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b, 'jax.Array')
    self.assertEqual(inferred.c, 'jax.Array')

  def testRound_ReturnsJaxArray(self):
    with utils.SaveCodeAsString() as code_saver:
      x = cast(jax.Array, jnp.zeros(3))
      y = jnp.round(x, 2)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, 'jax.Array')

  def testSum_ReturnsJaxArray(self):
    """Tests that jnp.sum(jax.Array) returns a jax.Array."""
    with utils.SaveCodeAsString() as code_saver:
      a = cast(jax.Array, jnp.zeros(3))
      # pylint: disable=unused-variable
      b = jnp.sum(a)
      c = jnp.sum(a, keepdims=True)
      d = jnp.sum(a, axis=0)
      e = jnp.sum(a, axis=0, keepdims=True)
      # pylint: enable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b, 'jax.Array')
    self.assertEqual(inferred.c, 'jax.Array')
    self.assertEqual(inferred.d, 'jax.Array')
    self.assertEqual(inferred.e, 'jax.Array')

  def testTranspsoe_ReturnsJaxArray(self):
    with utils.SaveCodeAsString() as code_saver:
      a = cast(jax.Array, jnp.zeros(3))
      b = jnp.transpose(a)  # pylint: disable=unused-variable
      c = jnp.transpose(a, axes=(0,))  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b, 'jax.Array')
    self.assertEqual(inferred.c, 'jax.Array')

  def testArray_HasDtypeAttribute(self):
    with utils.SaveCodeAsString() as code_saver:
      a = cast(jax.Array, jnp.zeros(3))
      b = a.dtype  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b, 'numpy.dtype')


class JnpNdarrayTests(absltest.TestCase):
  """Tests for operations on the plain jnp.ndarray class.

  Users might not have all their arrays being typed as Tensor Annotations
  types - they might also have some plain jnp.ndarrays around. We need to make
  sure they don't get weird type errors on those.
  """

  def testSlicingNdArray_ReturnsNdArray(self):
    with utils.SaveCodeAsString() as code_saver:
      a = cast(jnp.ndarray, jnp.zeros((2, 3)))
      b = a[0]  # pylint: disable=unused-variable
      c = a[0:1]  # pylint: disable=unused-variable
      d = a[:, 2:]  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.b, 'jax.numpy.ndarray')
    self.assertEqual(inferred.c, 'jax.numpy.ndarray')
    self.assertEqual(inferred.d, 'jax.numpy.ndarray')

if __name__ == '__main__':
  absltest.main()

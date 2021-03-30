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

from typing import Any, NewType

from absl.testing import absltest
import jax.numpy as jnp
from tensor_annotations import axes
from tensor_annotations.jax import Array0
from tensor_annotations.jax import Array1
from tensor_annotations.jax import Array2
from tensor_annotations.tests import utils


A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)

# It's less than ideal that we have to repeat imports etc. here for pytype, but
# this seems like the best balance between readability and complexity.
_PREAMBLE = """
from typing import NewType

import jax.numpy as jnp
from tensor_annotations import axes
from tensor_annotations.jax import Array0, Array1, Array2

A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
"""


class JAXStubTests(absltest.TestCase):

  def testTranspose_InferredShapeMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[A1, A2] = jnp.zeros((1, 2))
      y = jnp.transpose(x)

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, y.shape)

  def testUnaryOperator_ReturnCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[A1] = jnp.zeros((1,))
      # Let's just test a representative subset.
      a = jnp.abs(x)  # pylint: disable=unused-variable
      b = jnp.sin(x)  # pylint: disable=unused-variable
      c = jnp.floor(x)  # pylint: disable=unused-variable
      d = jnp.ones_like(x)  # pylint: disable=unused-variable
      e = jnp.round(x)  # pylint: disable=unused-variable
      f = jnp.sign(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    expected = 'Array1[A1]'
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

    self.assertEqual(inferred.a, 'Array0')
    self.assertEqual(inferred.b, 'Array0')
    self.assertEqual(inferred.c, 'Array1')
    self.assertEqual(inferred.d, 'Array1')
    self.assertEqual(inferred.e, 'Array2')
    self.assertEqual(inferred.f, 'Array2')

  def testSum_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[A1, A2] = jnp.zeros((1, 2))
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
      x: Array1[A1] = jnp.zeros((1,))
      a = jnp.sum(x, axis=0, keepdims=True)  # pylint: disable=unused-variable
      b = jnp.sum(x, keepdims=True)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Any')
    self.assertEqual(inferred.b, 'Any')

  def testTensorAdd_ReturnsCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array1[A1] = jnp.zeros((1,))
      a = x + 1  # pylint: disable=unused-variable
      b = x + x  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Array1[A1]', inferred.a)
    self.assertEqual('Array1[A1]', inferred.b)

  def testTensorUnaryOp_ReturnsCorrectTypeAndShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x1: Array0 = jnp.zeros(())
      y1 = abs(x1)  # pylint: disable=unused-variable
      y2 = -x1  # pylint: disable=unused-variable
      x2: Array1[A1] = jnp.zeros((1,))
      y3 = abs(x2)  # pylint: disable=unused-variable
      y4 = -x2  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Array0', inferred.y1)
    self.assertEqual('Array0', inferred.y2)
    self.assertEqual('Array1[A1]', inferred.y3)
    self.assertEqual('Array1[A1]', inferred.y4)

  def testBinaryOpWithScalar_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Array2[A1, A2] = jnp.zeros((1, 2))
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
      a: Array2[A1, A2] = jnp.zeros((1, 2))
      b: Array1[A2] = jnp.zeros((2,))
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
      a: Array2[A1, A2] = jnp.zeros((1, 2))
      b: Array2[A1, A2] = jnp.zeros((1, 2))
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


if __name__ == '__main__':
  absltest.main()

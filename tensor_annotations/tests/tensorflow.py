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

from typing import NewType
import unittest
from tensor_annotations import axes
from tensor_annotations.tensorflow import Tensor0
from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import Tensor2
from tensor_annotations.tests import utils
import tensorflow as tf


A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)

# It's less than ideal that we have to repeat imports etc. here for pytype, but
# this seems like the best balance between readability and complexity.
_PREAMBLE = """
from typing import NewType

import tensorflow as tf
from tensor_annotations import axes
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2

A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
"""


class TensorFlowStubTests(unittest.TestCase):

  def testTranspose_InferredMatchesActualShapeShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor2[A1, A2] = tf.zeros((1, 2))
      y = tf.transpose(x)

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y, y.shape)

  def testUnaryOperator_ReturnCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[A1] = tf.zeros((1,))
      # Let's just test a representative subset.
      a = tf.abs(x)  # pylint: disable=unused-variable
      b = tf.sin(x)  # pylint: disable=unused-variable
      c = tf.floor(x)  # pylint: disable=unused-variable
      d = tf.ones_like(x)  # pylint: disable=unused-variable
      e = tf.round(x)  # pylint: disable=unused-variable
      f = tf.sign(x)  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    expected = 'Tensor1[A1]'
    self.assertEqual(inferred.a, expected)
    self.assertEqual(inferred.b, expected)
    self.assertEqual(inferred.c, expected)
    self.assertEqual(inferred.d, expected)
    self.assertEqual(inferred.e, expected)
    self.assertEqual(inferred.f, expected)

  def testZerosOnes_ReturnsCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      a = tf.zeros((1,))  # pylint: disable=unused-variable
      b = tf.ones((1,))  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Tensor1')
    self.assertEqual(inferred.b, 'Tensor1')

  def testSum_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor2[A1, A2] = tf.zeros((1, 2))
      y1 = tf.reduce_sum(x, axis=0)
      y2 = tf.reduce_sum(x, axis=1)

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.y1, y1.shape)
    self.assertEqual(inferred.y2, y2.shape)

  def testTensorAdd_ReturnsCustomType(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor1[A1] = tf.zeros((1,))
      a = x + 1  # pylint: disable=unused-variable
      b = x + x  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual(inferred.a, 'Tensor1[A1]')
    self.assertEqual(inferred.b, 'Tensor1[A1]')

  def testTensorUnaryOp_ReturnsCorrectTypeAndShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x1: Tensor0 = tf.zeros(())
      y1 = abs(x1)  # pylint: disable=unused-variable
      y2 = -x1  # pylint: disable=unused-variable
      x2: Tensor1[A1] = tf.zeros((1,))
      y3 = abs(x2)  # pylint: disable=unused-variable
      y4 = -x2  # pylint: disable=unused-variable

    inferred = utils.pytype_infer_types(_PREAMBLE + code_saver.code)

    self.assertEqual('Tensor0', inferred.y1)
    self.assertEqual('Tensor0', inferred.y2)
    self.assertEqual('Tensor1[A1]', inferred.y3)
    self.assertEqual('Tensor1[A1]', inferred.y4)

  def testBinaryOpWithScalar_InferredMatchesActualShape(self):
    with utils.SaveCodeAsString() as code_saver:
      x: Tensor2[A1, A2] = tf.zeros((1, 2))
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
      a: Tensor2[A1, A2] = tf.zeros((1, 2))
      b: Tensor1[A2] = tf.zeros((2,))
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
      a: Tensor2[A1, A2] = tf.zeros((1, 2))
      b: Tensor2[A1, A2] = tf.zeros((1, 2))
      y1 = a + b
      y2 = a - b
      y3 = a / b
      y4 = a * b

    inferred = utils.pytype_infer_shapes(_PREAMBLE + code_saver.code)

    self.assertEqual(y1.shape, inferred.y1)
    self.assertEqual(y2.shape, inferred.y2)
    self.assertEqual(y3.shape, inferred.y3)
    self.assertEqual(y4.shape, inferred.y4)


if __name__ == '__main__':
  unittest.main()

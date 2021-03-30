"""Tests for experimental/decorators.py."""

from typing import List
import unittest

from tensor_annotations.axes import Height
from tensor_annotations.axes import Width
from tensor_annotations.experimental import decorators
from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import Tensor2
import tensorflow as tf


class DecoratorTests(unittest.TestCase):

  # ===== Tests for correct args. =====

  def test_correct_args_does_not_raise_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(
        x: Tensor1[Height],  # pylint: disable=unused-argument
        y: Tensor2[Height, Width],  # pylint: disable=unused-argument
    ):
      pass
    foo(tf.zeros([3]), tf.zeros([3, 5]))
    foo(x=tf.zeros([3]), y=tf.zeros([3, 5]))
    foo(tf.zeros([3]), y=tf.zeros([3, 5]))

  def test_correct_default_arg_does_not_raise_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(
        x: Tensor2[Height, Width] = tf.zeros([3, 5]),  # pylint: disable=unused-argument
    ):
      pass
    foo()
    foo(tf.zeros([3, 5]))

  # ===== Tests for unannotated args. =====

  def test_unannotated_arg_does_not_raise_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x):  # pylint: disable=unused-argument
      pass
    foo(tf.zeros([3]))
    foo(x=tf.zeros([3]))

  def test_unannotated_and_correct_annotated_arg_does_not_raise_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x, y: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    foo(tf.zeros([3]), tf.zeros([3, 5]))
    foo(x=tf.zeros([3]), y=tf.zeros([3, 5]))
    foo(tf.zeros([3]), y=tf.zeros([3, 5]))

  def test_unrelated_annotation_does_not_raise_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x: int):  # pylint: disable=unused-argument
      pass
    foo(0)

  def test_unrelated_generic_annotation_does_not_raise_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x: List[int]):  # pylint: disable=unused-argument
      pass
    foo([0])

  # ===== Tests for incorrect args. =====

  def test_incorrect_arg_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(ValueError):
      foo(tf.zeros([3]))
    with self.assertRaises(ValueError):
      foo(x=tf.zeros([3]))

  def test_incorrect_second_arg_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x, y: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(ValueError):
      foo(None, tf.zeros([3]))
    with self.assertRaises(ValueError):
      foo(x=None, y=tf.zeros([3]))
    with self.assertRaises(ValueError):
      foo(None, y=tf.zeros([3]))

  def test_incorrect_default_arg_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(
        x: Tensor2[Height, Width] = tf.zeros([3]),  # pylint: disable=unused-argument
    ):
      pass
    with self.assertRaises(ValueError):
      foo()

  def test_incorrect_overridden_default_arg_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(
        x: Tensor2[Height, Width] = tf.zeros([3, 5]),  # pylint: disable=unused-argument
    ):
      pass
    with self.assertRaises(ValueError):
      foo(tf.zeros([3]))
    with self.assertRaises(ValueError):
      foo(x=tf.zeros([3]))

  def test_non_tensor_arg_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(ValueError):
      foo(None)

  # ==== Make sure 'normal' errors still behave as expected. =====

  def test_no_args_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo(x: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(TypeError):
      foo()  # pylint: disable=no-value-for-parameter

  def test_too_many_args_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo():
      pass
    with self.assertRaises(TypeError):
      foo(None)  # pylint: disable=too-many-function-args

  def test_missing_keyword_arg_raises_exception(self):
    @decorators.verify_runtime_args_and_return_ranks
    def foo():
      pass
    with self.assertRaises(TypeError):
      foo(x=None)  # pylint: disable=unexpected-keyword-arg


if __name__ == '__main__':
  unittest.main()

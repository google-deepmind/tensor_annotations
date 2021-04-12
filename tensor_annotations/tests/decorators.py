"""Tests for experimental/decorators.py."""

from typing import Dict, List, NamedTuple, Tuple
import unittest

from tensor_annotations.axes import Height
from tensor_annotations.axes import Width
from tensor_annotations.experimental import decorators
from tensor_annotations.tensorflow import Tensor1
from tensor_annotations.tensorflow import Tensor2
import tensorflow as tf


class DecoratorsArgsTests(unittest.TestCase):

  # ===== Tests for correct args. =====

  def test_correct_args_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(
        x: Tensor1[Height],  # pylint: disable=unused-argument
        y: Tensor2[Height, Width],  # pylint: disable=unused-argument
    ):
      pass
    foo(tf.zeros([3]), tf.zeros([3, 5]))
    foo(x=tf.zeros([3]), y=tf.zeros([3, 5]))
    foo(tf.zeros([3]), y=tf.zeros([3, 5]))

  def test_correct_default_arg_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(
        x: Tensor2[Height, Width] = tf.zeros([3, 5]),  # pylint: disable=unused-argument
    ):
      pass
    foo()
    foo(tf.zeros([3, 5]))

  # ===== Tests for unannotated args. =====

  def test_unannotated_arg_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x):  # pylint: disable=unused-argument
      pass
    foo(tf.zeros([3]))
    foo(x=tf.zeros([3]))

  def test_unannotated_and_correct_annotated_arg_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x, y: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    foo(tf.zeros([3]), tf.zeros([3, 5]))
    foo(x=tf.zeros([3]), y=tf.zeros([3, 5]))
    foo(tf.zeros([3]), y=tf.zeros([3, 5]))

  def test_unrelated_annotation_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x: int):  # pylint: disable=unused-argument
      pass
    foo(0)

  def test_unrelated_generic_annotation_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x: List[int]):  # pylint: disable=unused-argument
      pass
    foo([0])

  # ===== Tests for incorrect args. =====

  def test_incorrect_arg_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(TypeError):
      foo(tf.zeros([3]))
    with self.assertRaises(TypeError):
      foo(x=tf.zeros([3]))

  def test_incorrect_second_arg_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x, y: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(TypeError):
      foo(None, tf.zeros([3]))
    with self.assertRaises(TypeError):
      foo(x=None, y=tf.zeros([3]))
    with self.assertRaises(TypeError):
      foo(None, y=tf.zeros([3]))

  def test_incorrect_default_arg_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(
        x: Tensor2[Height, Width] = tf.zeros([3]),  # pylint: disable=unused-argument
    ):
      pass
    with self.assertRaises(TypeError):
      foo()

  def test_incorrect_overridden_default_arg_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(
        x: Tensor2[Height, Width] = tf.zeros([3, 5]),  # pylint: disable=unused-argument
    ):
      pass
    with self.assertRaises(TypeError):
      foo(tf.zeros([3]))
    with self.assertRaises(TypeError):
      foo(x=tf.zeros([3]))

  def test_non_tensor_arg_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(TypeError):
      foo(None)

  # ==== Make sure 'normal' errors still behave as expected. =====

  def test_no_args_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo(x: Tensor2[Height, Width]):  # pylint: disable=unused-argument
      pass
    with self.assertRaises(TypeError):
      foo()  # pylint: disable=no-value-for-parameter

  def test_too_many_args_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo():
      pass
    with self.assertRaises(TypeError):
      foo(None)  # pylint: disable=too-many-function-args

  def test_missing_keyword_arg_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo():
      pass
    with self.assertRaises(TypeError):
      foo(x=None)  # pylint: disable=unexpected-keyword-arg


class DecoratorsReturnTests(unittest.TestCase):

  # ===== Cases which should *not* raise an exception. =====

  def test_return_value_is_preserved(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo():
      return 42
    self.assertEqual(foo(), 42)

  def test_correct_return_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo() -> Tensor2[Height, Width]:
      return tf.zeros([3, 5])
    foo()

  def test_non_tensor_annotation_does_not_raise_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo() -> None:
      # This is incorrect, but it's not our job to catch it.
      return tf.zeros([3])
    foo()

  # ===== Cases which *should* raise an exception. =====

  def test_incorrect_return_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo() -> Tensor2[Height, Width]:
      return tf.zeros([3])
    with self.assertRaises(TypeError):
      foo()

  def test_non_tensor_return_value_raises_exception(self):
    @decorators.verify_runtime_ranks_of_args_and_return
    def foo() -> Tensor2[Height, Width]:
      return None
    with self.assertRaises(TypeError):
      foo()


class DecoratorsTreeTests(unittest.TestCase):

  def test_convert_tuple_type_is_correct(self):
    tree_type = Tuple[int, str]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, (int, str))

  def test_convert_nested_tuple_type_is_correct(self):
    tree_type = Tuple[int, Tuple[float, str]]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, (int, (float, str)))

  def test_convert_double_nested_tuple_type_is_correct(self):
    tree_type = Tuple[int, Tuple[float, Tuple[str]]]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, (int, (float, (str,))))

  def test_convert_empty_tuple_type_is_correct(self):
    tree_type = Tuple[()]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, ())

  def test_convert_unparameterised_tuple_remains_unparameterised(self):
    tree_type = Tuple[str, Tuple]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, (str, Tuple))

  def test_convert_dict_type_is_correct(self):
    tree_type = Dict[str, int]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, {str: int})

  def test_convert_nested_dict_type_is_correct(self):
    tree_type = Dict[str, Dict[int, float]]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, {str: {int: float}})

  def test_convert_double_nested_dict_type_is_correct(self):
    tree_type = Dict[str, Dict[bool, Dict[int, float]]]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, {str: {bool: {int: float}}})

  def test_convert_unparameterised_dict_remains_unparameterised(self):
    tree_type = Tuple[Dict]
    type_tree = decorators._tree_type_to_type_tree(tree_type)
    self.assertEqual(type_tree, (Dict,))

  def test_convert_named_tuple_type_is_correct(self):
    class T(NamedTuple):
      a: Tuple[int]
      b: Dict[str, bool]
    type_tree = decorators._tree_type_to_type_tree(T)
    self.assertEqual(type_tree, T(a=(int,), b={str: bool}))

  def test_convert_nested_named_tuple_type_is_correct(self):
    class T1(NamedTuple):
      a: Tuple[int]
    class T2(NamedTuple):
      b: T1
    type_tree = decorators._tree_type_to_type_tree(T2)
    self.assertEqual(
        type_tree,
        T2(
            b=T1(
                a=(int,),
            ),
        ),
    )

  def test_convert_double_nested_named_tuple_type_is_correct(self):
    class T1(NamedTuple):
      a: Tuple[int]
    class T2(NamedTuple):
      b: T1
    class T3(NamedTuple):
      c: T2
    type_tree = decorators._tree_type_to_type_tree(T3)
    self.assertEqual(
        type_tree,
        T3(
            c=T2(
                b=T1(
                    a=(int,),
                )
            ),
        ),
    )

  def test_convert_mixed_type_is_correct(self):
    class Trajectory(NamedTuple):
      actions: Dict[str, Tuple[float]]
      observations: Dict[str, Tuple[float]]

    class TrajectoriesAndParameters(NamedTuple):
      trajectories: Tuple[Trajectory]
      parameters: Tuple[float]

    type_tree = decorators._tree_type_to_type_tree(TrajectoriesAndParameters)
    self.assertEqual(
        type_tree,
        TrajectoriesAndParameters(
            trajectories=(
                Trajectory(
                    actions={str: (float,)},
                    observations={str: (float,)},
                ),
            ),
            parameters=(float,),
        )
    )


if __name__ == '__main__':
  unittest.main()

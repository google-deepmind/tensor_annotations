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
"""Custom tensor classes for JAX supporting shape parameterisation.

Note that these should only be used for the purposes of type annotation and
should never be instantiated. (Certain IDEs may also use these for
autocompletion, too.)

Type annotations for these classes are maintained in a separate stubs file,
`tensorflow.pyi`.
"""

# LINT.IfChange

from typing import Any, Generic, TypeVar

from tensor_annotations import axes

A1 = TypeVar('A1', bound=axes.Axis)
A2 = TypeVar('A2', bound=axes.Axis)
A3 = TypeVar('A3', bound=axes.Axis)
A4 = TypeVar('A4', bound=axes.Axis)
A5 = TypeVar('A5', bound=axes.Axis)
A6 = TypeVar('A6', bound=axes.Axis)
A7 = TypeVar('A7', bound=axes.Axis)
A8 = TypeVar('A8', bound=axes.Axis)


# We need to define DTypes ourselves rather than use e.g. tf.uint8 because
# according to typing.py, tf.uint8 etc aren't actually types, so they can't
# be used as type arguments.
# pylint: disable=invalid-name,multiple-statements,g-wrong-blank-lines
class DType: pass
class uint8(DType): pass
class uint16(DType): pass
class uint32(DType): pass
class uint64(DType): pass
class int8(DType): pass
class int16(DType): pass
class int32(DType): pass
class int64(DType): pass
class float16(DType): pass
class float32(DType): pass
class float64(DType): pass
class complex64(DType): pass
class complex128(DType): pass
class bfloat16(DType): pass
# Yup, these two definitely are native dtypes in TensorFlow:
# https://www.tensorflow.org/api_docs/python/tf/dtypes
class string(DType): pass
# TensorFlow's boolean dtype is definitely just 'bool'.
# It's a little annoying that has the same name as the Python keyword,
# but let's stick with TensorFlow's naming.
class bool(DType): pass  # pylint: disable=redefined-builtin
# pylint: enable=invalid-name, multiple-statements,g-wrong-blank-lines

# We want to have an `AnyDType` type that behaves like `Any` but for DTypes.
#
# Should `AnyDType` just be the parent class `DType` itself? No. Consider the
# following example:
#
#     def takes_specific_type(x: uint8): ...
#     def returns_nonspecific_type() -> DType: ...
#     y = returns_nonspecific_type()
#     foo(y)
#
# This doesn't type-check correctly. `DType` cannot be used in place of the
# more specific type `uint8`. We want our `AnyDType` type to have the property
# that it can be used *anywhere* - including as an argument to a function that
# takes a specific type. So using `DType` as our `AnyDType` won't work.
#
# What about a union of the dtypes above? Initially I thought no.
# Consider the following example:
#
#     def takes_specific_type(x: uint8): ...
#     y: Union[uint8, uint16]
#     foo(y)
#
# I *thought* this would be a type error, because we can't guarantee that
# `y` is definitely uint8, but it turns out that both mypy and pytype are fine
# with it.
#
# But anyway, we can't do a union of the above types for another reason:
# pytype breaks if we do a union of too many types.
#
# So in the end, we just set this to be an alias of `Any`, so the meaning is
# clearer in code. Unfortunately, it still shows up as `Any` in pytype output.
# But hey, it's the best we can do.
AnyDType = Any

DT = TypeVar('DT', bound=DType)


class _TensorBase:
  """Base class for TensorN classes containing common methods and attributes."""

  def __new__(cls, *args, **kwargs):
    raise TypeError('tensor_annotations tensors should not be instantiated')

  # These are necessary so that type checkers know we have these methods.
  __abs__: Any
  __add__: Any
  __add__: Any
  __float__: Any
  __floordiv__: Any
  __ge__: Any
  __gt__: Any
  __le__: Any
  __len__: Any
  __lt__: Any
  __matmul__: Any
  __mul__: Any
  __neg__: Any
  __neg__: Any
  __pos__: Any
  __pow__: Any
  __rmatmul__: Any
  __rmul__: Any
  __sub__: Any
  __truediv__: Any
  shape: Any
  dtype: Any

  def numpy(self) -> Any:
    pass


class Tensor0(Generic[DT], _TensorBase):
  """A scalar - produced by e.g. tf.reduce_sum(tf.zeros((2, 3)))."""
  pass


class Tensor1(Generic[DT, A1], _TensorBase):
  """A tensor of rank 1."""
  pass


class Tensor2(Generic[DT, A1, A2], _TensorBase):
  """A tensor of rank 2."""
  pass


class Tensor3(Generic[DT, A1, A2, A3], _TensorBase):
  """A tensor of rank 3."""
  pass


class Tensor4(Generic[DT, A1, A2, A3, A4], _TensorBase):
  """A tensor of rank 4."""
  pass


class Tensor5(Generic[DT, A1, A2, A3, A4, A5], _TensorBase):
  """A tensor of rank 5."""
  pass


class Tensor6(Generic[DT, A1, A2, A3, A4, A5, A6], _TensorBase):
  """A tensor of rank 6."""
  pass


class Tensor7(Generic[DT, A1, A2, A3, A4, A5, A6, A7], _TensorBase):
  """A tensor of rank 7."""
  pass


class Tensor8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8], _TensorBase):
  """A tensor of rank 8."""
  pass


Tensor0AnyDType = Tensor0[AnyDType]
Tensor1AnyDType = Tensor1[AnyDType, A1]
Tensor2AnyDType = Tensor2[AnyDType, A1, A2]
Tensor3AnyDType = Tensor3[AnyDType, A1, A2, A3]
Tensor4AnyDType = Tensor4[AnyDType, A1, A2, A3, A4]
Tensor5AnyDType = Tensor5[AnyDType, A1, A2, A3, A4, A5]
Tensor6AnyDType = Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]
Tensor7AnyDType = Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]
Tensor8AnyDType = Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]

# LINT.ThenChange(tensorflow.pyi)

# LINT.IfChange
"""Type stubs for custom TensorFlow tensor classes.

NOTE: This file is generated from templates/tensorflow_tensors.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_tensor_template.py \
     --template templates/tensorflow_tensors.pyi \
     --out tensorflow.pyi
"""

from typing import Any, Literal, TypeVar, Tuple, Sequence, Generic, overload, Union

import numpy as np
import tensorflow as tf
from tensor_annotations.axes import Axis


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)
A5 = TypeVar('A5', bound=Axis)
A6 = TypeVar('A6', bound=Axis)
A7 = TypeVar('A7', bound=Axis)
A8 = TypeVar('A8', bound=Axis)


# We need to define DTypes ourselves rather than use e.g. tf.uint8 because
# according to typing.py, tf.uint8 etc aren't actually types, so they can't
# be used as type arguments.
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
class bool(DType): pass

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

Number = Union[int, float]

{% set unary_funcs = ['__abs__', '__neg__', '__pos__'] %}
{# Yes, __mul__ _is_ elementwise! __matmul__ is matrix multiply. #}
{% set binary_elementwise_funcs = ['__add__', '__sub__', '__floordiv__',
                                   '__truediv__', '__pow__', '__lt__', '__le__',
                                   '__ge__', '__gt__', '__mul__', '__rmul__'] %}


# A quick refresher on broadcasting rules:
# 1. Tensor[A, B] + scalar = Tensor[A, B]
# 2. Otherwise, start with trailing dimension of each tensor and work
#    forwards. Broadcasting is possible if, for each axis, the dimensions
#    of that axis in each tensor are either a) equal or b) one of them is 1.
# We deliberately ignore case b) for the time being since we don't support
# literal shapes yet.

class Tensor0(Generic[DT]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # Technically, arrays of any rank can float()ed if they only contain a
  # single value, but we can only guarantee it for Tensor0.
  def __float__(self) -> float: ...

  def numpy(self) -> Any: ...  # Returns a scalar value, *not* an ndarray.
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor0[DT]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor0[AnyDType]: ...
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A1]) -> Tensor1[AnyDType, A1]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A1, A2]) -> Tensor2[AnyDType, A1, A2]: ...
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A1, A2, A3]) -> Tensor3[AnyDType, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Tensor4[AnyDType, A1, A2, A3, A4]) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor5[AnyDType, A1, A2, A3, A4, A5]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  {% endfor %}
  # END: Binary element-wise operators


class Tensor1(Generic[DT, A1]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor1[DT, A1]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor1[AnyDType, A1]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor1[AnyDType, A1]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A1]) -> Tensor1[AnyDType, A1]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor2(Generic[DT, A1, A2]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor2[DT, A1, A2]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor2[AnyDType, A1, A2]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor2[AnyDType, A1, A2]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A2]) -> Tensor2[AnyDType, A1, A2]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A1, A2]) -> Tensor2[AnyDType, A1, A2]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Tensor2[AnyDType, A2, A3]) -> Tensor2[AnyDType, A1, A3]: ...

  @overload
  def __matmul__(self, other: Tensor3[AnyDType, A4, A2, A3]) -> Tensor3[AnyDType, A4, A1, A3]: ...

  @overload
  def __matmul__(self, other: Tensor4[AnyDType, A4, A5, A2, A3]) -> Tensor4[AnyDType, A4, A5, A1, A3]: ...

  @overload
  def __rmatmul__(self, other: Tensor2[AnyDType, A3, A1]) -> Tensor2[AnyDType, A3, A2]: ...

  @overload
  def __rmatmul__(self, other: Tensor3[AnyDType, A3, A4, A1]) -> Tensor3[AnyDType, A3, A4, A2]: ...

  @overload
  def __rmatmul__(self, other: Tensor4[AnyDType, A3, A4, A5, A1]) -> Tensor4[AnyDType, A3, A4, A5, A2]: ...
  # END: The `@` operator


class Tensor3(Generic[DT, A1, A2, A3]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor3[DT, A1, A2, A3]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor3[AnyDType, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor3[AnyDType, A1, A2, A3]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A3]) -> Tensor3[AnyDType, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A2, A3]) -> Tensor3[AnyDType, A1, A2, A3]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A1, A2, A3]) -> Tensor3[AnyDType, A1, A2, A3]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Tensor2[AnyDType, A3, A4]) -> Tensor3[AnyDType, A1, A2, A4]: ...

  @overload
  def __matmul__(self, other: Tensor3[AnyDType, A1, A3, A4]) -> Tensor3[AnyDType, A1, A2, A4]: ...

  @overload
  def __matmul__(self, other: Tensor4[AnyDType, A5, A1, A3, A4]) -> Tensor4[AnyDType, A5, A1, A2, A4]: ...

  @overload
  def __rmatmul__(self, other: Tensor2[AnyDType, A4, A2]) -> Tensor3[AnyDType, A1, A4, A3]: ...

  @overload
  def __rmatmul__(self, other: Tensor3[AnyDType, A1, A4, A2]) -> Tensor3[AnyDType, A1, A4, A3]: ...

  @overload
  def __rmatmul__(self, other: Tensor4[AnyDType, A5, A1, A4, A2]) -> Tensor4[AnyDType, A5, A1, A4, A3]: ...
  # END: The `@` operator


class Tensor4(Generic[DT, A1, A2, A3, A4]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor4[DT, A1, A2, A3, A4]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A4]) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A3, A4]) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A2, A3, A4]) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor4[AnyDType, A1, A2, A3, A4]) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Tensor2[AnyDType, A4, A5]) -> Tensor4[AnyDType, A1, A2, A3, A5]: ...

  @overload
  def __matmul__(self, other: Tensor3[AnyDType, A2, A4, A5]) -> Tensor4[AnyDType, A1, A2, A3, A5]: ...

  @overload
  def __matmul__(self, other: Tensor4[AnyDType, A1, A2, A4, A5]) -> Tensor4[AnyDType, A1, A2, A3, A5]: ...

  @overload
  def __rmatmul__(self, other: Tensor2[AnyDType, A5, A3]) -> Tensor4[AnyDType, A1, A2, A5, A4]: ...

  @overload
  def __rmatmul__(self, other: Tensor3[AnyDType, A2, A5, A3]) -> Tensor4[AnyDType, A1, A2, A5, A4]: ...

  @overload
  def __rmatmul__(self, other: Tensor4[AnyDType, A1, A2, A5, A3]) -> Tensor4[AnyDType, A1, A2, A5, A4]: ...
  # END: The `@` operator


class Tensor5(Generic[DT, A1, A2, A3, A4, A5]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A5]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A4, A5]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A3, A4, A5]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor4[AnyDType, A2, A3, A4, A5]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor5[AnyDType, A1, A2, A3, A4, A5]) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor6(Generic[DT, A1, A2, A3, A4, A5, A6]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A5, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A4, A5, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor4[AnyDType, A3, A4, A5, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor5[AnyDType, A2, A3, A4, A5, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor7(Generic[DT, A1, A2, A3, A4, A5, A6, A7]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A5, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor4[AnyDType, A4, A5, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor5[AnyDType, A3, A4, A5, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType
  def __len__(self) -> int: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor0[AnyDType]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[AnyDType, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor2[AnyDType, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor3[AnyDType, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor4[AnyDType, A5, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor5[AnyDType, A4, A5, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {% endfor %}

  # END: Binary element-wise operators


Tensor0AnyDType = Tensor0[AnyDType]
Tensor1AnyDType = Tensor1[AnyDType, A1]
Tensor2AnyDType = Tensor2[AnyDType, A1, A2]
Tensor3AnyDType = Tensor3[AnyDType, A1, A2, A3]
Tensor4AnyDType = Tensor4[AnyDType, A1, A2, A3, A4]
Tensor5AnyDType = Tensor5[AnyDType, A1, A2, A3, A4, A5]
Tensor6AnyDType = Tensor6[AnyDType, A1, A2, A3, A4, A5, A6]
Tensor7AnyDType = Tensor7[AnyDType, A1, A2, A3, A4, A5, A6, A7]
Tensor8AnyDType = Tensor8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]
# LINT.ThenChange(../tensorflow.pyi)

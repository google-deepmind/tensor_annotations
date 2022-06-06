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
DT = TypeVar('DT')

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
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor0[DT]: ...
  @overload
  def {{ func }}(self, other: Tensor1[DT, A1]) -> Tensor1[DT, A1]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A1, A2]) -> Tensor2[DT, A1, A2]: ...
  @overload
  def {{ func }}(self, other: Tensor3[DT, A1, A2, A3]) -> Tensor3[DT, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Tensor4[DT, A1, A2, A3, A4]) -> Tensor4[DT, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor5[DT, A1, A2, A3, A4, A5]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor6[DT, A1, A2, A3, A4, A5, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  {% endfor %}
  # END: Binary element-wise operators


class Tensor1(Generic[DT, A1]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor1[DT, A1]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor1[DT, A1]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor1[DT, A1]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A1]) -> Tensor1[DT, A1]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor2(Generic[DT, A1, A2]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor2[DT, A1, A2]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor2[DT, A1, A2]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor2[DT, A1, A2]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A2]) -> Tensor2[DT, A1, A2]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor2[DT, A1, A2]) -> Tensor2[DT, A1, A2]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor3(Generic[DT, A1, A2, A3]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor3[DT, A1, A2, A3]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor3[DT, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor3[DT, A1, A2, A3]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A3]) -> Tensor3[DT, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A2, A3]) -> Tensor3[DT, A1, A2, A3]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor3[DT, A1, A2, A3]) -> Tensor3[DT, A1, A2, A3]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor4(Generic[DT, A1, A2, A3, A4]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor4[DT, A1, A2, A3, A4]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor4[DT, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor4[DT, A1, A2, A3, A4]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A4]) -> Tensor4[DT, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A3, A4]) -> Tensor4[DT, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Tensor3[DT, A2, A3, A4]) -> Tensor4[DT, A1, A2, A3, A4]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor4[DT, A1, A2, A3, A4]) -> Tensor4[DT, A1, A2, A3, A4]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor5(Generic[DT, A1, A2, A3, A4, A5]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A5]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A4, A5]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor3[DT, A3, A4, A5]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Tensor4[DT, A2, A3, A4, A5]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor5[DT, A1, A2, A3, A4, A5]) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor6(Generic[DT, A1, A2, A3, A4, A5, A6]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A5, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor3[DT, A4, A5, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor4[DT, A3, A4, A5, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Tensor5[DT, A2, A3, A4, A5, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor6[DT, A1, A2, A3, A4, A5, A6]) -> Tensor6[DT, A1, A2, A3, A4, A5, A6]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor7(Generic[DT, A1, A2, A3, A4, A5, A6, A7]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor3[DT, A5, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor4[DT, A4, A5, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor5[DT, A3, A4, A5, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Tensor6[DT, A2, A3, A4, A5, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]) -> Tensor7[DT, A1, A2, A3, A4, A5, A6, A7]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Tensor8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  def numpy(self) -> np.ndarray: ...
  shape: tf.TensorShape
  dtype: tf.DType

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor0[DT]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Tensor1[DT, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor2[DT, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor3[DT, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor4[DT, A5, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor5[DT, A4, A5, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor6[DT, A3, A4, A5, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Tensor7[DT, A2, A3, A4, A5, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]) -> Tensor8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {% endfor %}

  # END: Binary element-wise operators

# LINT.ThenChange(../tensorflow.pyi)

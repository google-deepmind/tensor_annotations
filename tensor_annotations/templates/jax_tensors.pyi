# LINT.IfChange
"""Type stubs for custom JAX tensor classes.

NOTE: This file is generated from templates/jax_tensors.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_tensor_template.py \
     --template templates/jax_tensors.pyi \
     --out jax.pyi
"""

from typing import Any, TypeVar, Tuple, Sequence, Generic, overload, Union

from tensor_annotations.axes import Axis


Shape = Sequence[int]
Shape1 = Tuple[int]
Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Shape4 = Tuple[int, int, int, int]

A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)

Number = Union[int, float]

{% set unary_funcs = ['__abs__', '__neg__', '__pos__'] %}
{# Yes, __mul__ _is_ elementwise! __matmul__ is matrix multiply. #}
{% set binary_elementwise_funcs = ['__add__', '__sub__', '__floordiv__',
                                   '__truediv__', '__pow__', '__lt__', '__le__',
                                   '__ge__', '__gt__', '__mul__', '__rmul__'] %}


# A quick refresher on broadcasting rules:
# 1. array[A, B] + scalar = array[A, B].
# 2. Otherwise, start with trailing dimension of each tensor and work
#    forwards. Broadcasting is possible if, for each axis, the dimensions
#    of that axis in each tensor are either a) equal or b) one of them is 1.
# We deliberately ignore case b) for the time being since we don't support
# literal shapes yet.

class array0:
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> array0: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: array0) -> array0: ...
  @overload
  def {{ func }}(self, other: array1) -> array1: ...
  @overload
  def {{ func }}(self, other: array2) -> array2: ...
  @overload
  def {{ func }}(self, other: array3) -> array3: ...
  @overload
  def {{ func }}(self, other: array4) -> array4: ...
  {% endfor %}
  # END: Binary element-wise operators


class array1(Generic[A1]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> array1[A1]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  # Broadcasting case 1: array[A, B] + scalar = array[A, B].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Number) -> array1[A1]: ...
  @overload
  def {{ func }}(self, other: array0) -> array1[A1]: ...
  {% endfor %}

  # Broadcasting case 2: array[A, B, C] + array[B, C] = array[A, B, C].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: array1[A1]) -> array1[A1]: ...
  {% endfor %}

  # END: Binary element-wise operators


class array2(Generic[A1, A2]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> array2[A1, A2]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  # Broadcasting case 1: array[A, B] + scalar = array[A, B].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Number) -> array2[A1, A2]: ...
  @overload
  def {{ func }}(self, other: array0) -> array2[A1, A2]: ...
  {% endfor %}

  # Broadcasting case 2: array[A, B, C] + array[B, C] = array[A, B, C].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: array1[A2]) -> array2[A1, A2]: ...
  @overload
  def {{ func }}(self, other: array2[A1, A2]) -> array2[A1, A2]: ...
  {% endfor %}

  # END: Binary element-wise operators


class array3(Generic[A1, A2, A3]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> array3[A1, A2, A3]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  # Broadcasting case 1: array[A, B] + scalar = array[A, B].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Number) -> array3[A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: array0) -> array3[A1, A2, A3]: ...
  {% endfor %}

  # Broadcasting case 2: array[A, B, C] + array[B, C] = array[A, B, C].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: array1[A3]) -> array3[A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: array2[A2, A3]) -> array3[A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: array3[A1, A2, A3]) -> array3[A1, A2, A3]: ...
  {% endfor %}

  # END: Binary element-wise operators


class array4(Generic[A1, A2, A3, A4]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> array4[A1, A2, A3, A4]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  # Broadcasting case 1: array[A, B] + scalar = array[A, B].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Number) -> array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: array0) -> array4[A1, A2, A3, A4]: ...
  {% endfor %}

  # Broadcasting case 2: array[A, B, C] + array[B, C] = array[A, B, C].
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: array1[A4]) -> array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: array2[A3, A4]) -> array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: array3[A2, A3, A4]) -> array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: array4[A1, A2, A3, A4]) -> array4[A1, A2, A3, A4]: ...
  {% endfor %}

  # END: Binary element-wise operators

# LINT.ThenChange(../jax.pyi)

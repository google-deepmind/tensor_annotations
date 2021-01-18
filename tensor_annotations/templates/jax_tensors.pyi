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
# 1. Array[A, B] + scalar = Array[A, B].
# 2. Otherwise, start with trailing dimension of each tensor and work
#    forwards. Broadcasting is possible if, for each axis, the dimensions
#    of that axis in each tensor are either a) equal or b) one of them is 1.
# We deliberately ignore case b) for the time being since we don't support
# literal shapes yet.

class Array0:
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array0: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Array0) -> Array0: ...
  @overload
  def {{ func }}(self, other: Array1) -> Array1: ...
  @overload
  def {{ func }}(self, other: Array2) -> Array2: ...
  @overload
  def {{ func }}(self, other: Array3) -> Array3: ...
  @overload
  def {{ func }}(self, other: Array4) -> Array4: ...
  {% endfor %}
  # END: Binary element-wise operators


class Array1(Generic[A1]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array1[A1]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array1[A1]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array1[A1]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array1[A1]) -> Array1[A1]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array2(Generic[A1, A2]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array2[A1, A2]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array2[A1, A2]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array2[A1, A2]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A2]) -> Array2[A1, A2]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array2[A1, A2]) -> Array2[A1, A2]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array3(Generic[A1, A2, A3]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array3[A1, A2, A3]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array3[A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array3[A1, A2, A3]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A3]) -> Array3[A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Array2[A2, A3]) -> Array3[A1, A2, A3]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array3[A1, A2, A3]) -> Array3[A1, A2, A3]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array4(Generic[A1, A2, A3, A4]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array4[A1, A2, A3, A4]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array4[A1, A2, A3, A4]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A4]) -> Array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Array2[A3, A4]) -> Array4[A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Array3[A2, A3, A4]) -> Array4[A1, A2, A3, A4]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array4[A1, A2, A3, A4]) -> Array4[A1, A2, A3, A4]: ...

  {% endfor %}

  # END: Binary element-wise operators

# LINT.ThenChange(../jax.pyi)

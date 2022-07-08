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
A5 = TypeVar('A5', bound=Axis)
A6 = TypeVar('A6', bound=Axis)
A7 = TypeVar('A7', bound=Axis)
A8 = TypeVar('A8', bound=Axis)

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
  shape: Tuple[()]

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
  shape: Tuple[int]

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
  shape: Tuple[int, int]

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
  shape: Tuple[int, int, int]

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
  shape: Tuple[int, int, int, int]

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


class Array5(Generic[A1, A2, A3, A4, A5]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array5[A1, A2, A3, A4, A5]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array5[A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array5[A1, A2, A3, A4, A5]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A5]) -> Array5[A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array2[A4, A5]) -> Array5[A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array3[A3, A4, A5]) -> Array5[A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array4[A2, A3, A4, A5]) -> Array5[A1, A2, A3, A4, A5]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array5[A1, A2, A3, A4, A5]) -> Array5[A1, A2, A3, A4, A5]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array6(Generic[A1, A2, A3, A4, A5, A6]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array6[A1, A2, A3, A4, A5, A6]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array6[A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array6[A1, A2, A3, A4, A5, A6]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A6]) -> Array6[A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array2[A5, A6]) -> Array6[A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array3[A4, A5, A6]) -> Array6[A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array4[A3, A4, A5, A6]) -> Array6[A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array5[A2, A3, A4, A5, A6]) -> Array6[A1, A2, A3, A4, A5, A6]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array6[A1, A2, A3, A4, A5, A6]) -> Array6[A1, A2, A3, A4, A5, A6]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array7(Generic[A1, A2, A3, A4, A5, A6, A7]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array2[A6, A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array3[A5, A6, A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array4[A4, A5, A6, A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array5[A3, A4, A5, A6, A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array6[A2, A3, A4, A5, A6, A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array7[A1, A2, A3, A4, A5, A6, A7]) -> Array7[A1, A2, A3, A4, A5, A6, A7]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array8(Generic[A1, A2, A3, A4, A5, A6, A7, A8]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array0) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array2[A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array3[A6, A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array4[A5, A6, A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array5[A4, A5, A6, A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array6[A3, A4, A5, A6, A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array7[A2, A3, A4, A5, A6, A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array8[A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {% endfor %}

  # END: Binary element-wise operators


Array0AnyDType = Array0
Array1AnyDType = Array1
Array2AnyDType = Array2
Array3AnyDType = Array3
Array4AnyDType = Array4
Array5AnyDType = Array5
Array6AnyDType = Array6
Array7AnyDType = Array7
Array8AnyDType = Array8
# LINT.ThenChange(../jax.pyi)

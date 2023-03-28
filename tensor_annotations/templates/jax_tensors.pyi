# LINT.IfChange
"""Type stubs for custom JAX tensor classes.

NOTE: This file is generated from templates/jax_tensors.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_tensor_template.py \
     --template templates/jax_tensors.pyi \
     --out jax.pyi
"""

from typing import Any, TypeVar, Tuple, Sequence, Generic, overload, Union, Literal

from tensor_annotations.axes import Axis


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)
A5 = TypeVar('A5', bound=Axis)
A6 = TypeVar('A6', bound=Axis)
A7 = TypeVar('A7', bound=Axis)
A8 = TypeVar('A8', bound=Axis)

# We need to define DTypes ourselves rather than use e.g. jnp.uint8 because
# pytype sees JAX's own DTypes as `Any``.
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
class bfloat16(DType): pass

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
                                   '__ge__', '__gt__', '__eq__', '__ne__',
                                   '__mul__', '__rmul__'] %}


# A quick refresher on broadcasting rules:
# 1. Array[DT, A, B] + scalar = Array[DT, A, B].
# 2. Otherwise, start with trailing dimension of each tensor and work
#    forwards. Broadcasting is possible if, for each axis, the dimensions
#    of that axis in each tensor are either a) equal or b) one of them is 1.
# We deliberately ignore case b) for the time being since we don't support
# literal shapes yet.

class Array0(Generic[DT]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # Technically, arrays of any rank can float()ed if they only contain a
  # single value, but we can only guarantee it for Array0.
  def __float__(self) -> float: ...

  shape: Tuple[()]
  T: Array0[DT]
  at: Any
  ndim: Literal[0]
  dtype: type
  def astype(self, dtype) -> Array0[AnyDType]: ...

  # Technically this exists on all instances of JAX arrays,
  # but it throws an error for anything apart from a scalar
  # array, eg jnp.array(0).
  def item(self) -> Union[int, float, bool, complex]: ...


  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array0[DT]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators
  {% for func in binary_elementwise_funcs %}
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  {% endfor %}
  # END: Binary element-wise operators


class Array1(Generic[DT, A1]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int]
  T: Array1[DT, A1]
  at: Any
  ndim: Literal[1]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array1[AnyDType, A1]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array1[DT, A1]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array1[AnyDType, A1]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array1[AnyDType, A1]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Array1[AnyDType, A1]) -> Array0[AnyDType]: ...

  @overload
  def __matmul__(self, other: Array2[AnyDType, A1, A2]) -> Array1[AnyDType, A2]: ...

  @overload
  def __matmul__(self, other: Array3[AnyDType, A3, A1, A2]) -> Array2[AnyDType, A3, A2]: ...

  @overload
  def __matmul__(self, other: Array4[AnyDType, A3, A4, A1, A2]) -> Array3[AnyDType, A3, A4, A2]: ...

  @overload
  def __rmatmul__(self, other: Array1[AnyDType, A1]) -> Array0[AnyDType]: ...

  @overload
  def __rmatmul__(self, other: Array2[AnyDType, A2, A1]) -> Array1[AnyDType, A2]: ...

  @overload
  def __rmatmul__(self, other: Array3[AnyDType, A3, A2, A1]) -> Array2[AnyDType, A3, A2]: ...

  @overload
  def __rmatmul__(self, other: Array4[AnyDType, A3, A4, A2, A1]) -> Array3[AnyDType, A3, A4, A2]: ...
  # END: The `@` operator


class Array2(Generic[DT, A1, A2]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int]
  T: Array2[DT, A2, A1]
  at: Any
  ndim: Literal[2]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array2[AnyDType, A1, A2]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array2[DT, A1, A2]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array2[AnyDType, A1, A2]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Array1[AnyDType, A2]) -> Array1[AnyDType, A1]: ...

  @overload
  def __matmul__(self, other: Array2[AnyDType, A2, A3]) -> Array2[AnyDType, A1, A3]: ...

  @overload
  def __matmul__(self, other: Array3[AnyDType, A4, A2, A3]) -> Array3[AnyDType, A4, A1, A3]: ...

  @overload
  def __matmul__(self, other: Array4[AnyDType, A4, A5, A2, A3]) -> Array4[AnyDType, A4, A5, A1, A3]: ...

  @overload
  def __rmatmul__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A2]: ...

  @overload
  def __rmatmul__(self, other: Array2[AnyDType, A3, A1]) -> Array2[AnyDType, A3, A2]: ...

  @overload
  def __rmatmul__(self, other: Array3[AnyDType, A3, A4, A1]) -> Array3[AnyDType, A3, A4, A2]: ...

  @overload
  def __rmatmul__(self, other: Array4[AnyDType, A3, A4, A5, A1]) -> Array4[AnyDType, A3, A4, A5, A2]: ...
  # END: The `@` operator


class Array3(Generic[DT, A1, A2, A3]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int]
  T: Array3[DT, A3, A2, A1]
  at: Any
  ndim: Literal[3]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array3[AnyDType, A1, A2, A3]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array3[DT, A1, A2, A3]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array3[AnyDType, A1, A2, A3]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Array1[AnyDType, A3]) -> Array2[AnyDType, A1, A2]: ...

  @overload
  def __matmul__(self, other: Array2[AnyDType, A3, A4]) -> Array3[AnyDType, A1, A2, A4]: ...

  @overload
  def __matmul__(self, other: Array3[AnyDType, A1, A3, A4]) -> Array3[AnyDType, A1, A2, A4]: ...

  @overload
  def __matmul__(self, other: Array4[AnyDType, A5, A1, A3, A4]) -> Array4[AnyDType, A5, A1, A2, A4]: ...

  @overload
  def __rmatmul__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A3]: ...

  @overload
  def __rmatmul__(self, other: Array2[AnyDType, A4, A2]) -> Array3[AnyDType, A1, A4, A3]: ...

  @overload
  def __rmatmul__(self, other: Array3[AnyDType, A1, A4, A2]) -> Array3[AnyDType, A1, A4, A3]: ...

  @overload
  def __rmatmul__(self, other: Array4[AnyDType, A5, A1, A4, A2]) -> Array4[AnyDType, A5, A1, A4, A3]: ...
  # END: The `@` operator


class Array4(Generic[DT, A1, A2, A3, A4]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]
  T: Array4[DT, A4, A3, A2, A1]
  at: Any
  ndim: Literal[4]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array4[DT, A1, A2, A3, A4]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  {% endfor %}

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Array1[AnyDType, A4]) -> Array3[AnyDType, A1, A2, A3]: ...

  @overload
  def __matmul__(self, other: Array2[AnyDType, A4, A5]) -> Array4[AnyDType, A1, A2, A3, A5]: ...

  @overload
  def __matmul__(self, other: Array3[AnyDType, A2, A4, A5]) -> Array4[AnyDType, A1, A2, A3, A5]: ...

  @overload
  def __matmul__(self, other: Array4[AnyDType, A1, A2, A4, A5]) -> Array4[AnyDType, A1, A2, A3, A5]: ...

  @overload
  def __rmatmul__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A4]: ...

  @overload
  def __rmatmul__(self, other: Array2[AnyDType, A5, A3]) -> Array4[AnyDType, A1, A2, A5, A4]: ...

  @overload
  def __rmatmul__(self, other: Array3[AnyDType, A2, A5, A3]) -> Array4[AnyDType, A1, A2, A5, A4]: ...

  @overload
  def __rmatmul__(self, other: Array4[AnyDType, A1, A2, A5, A3]) -> Array4[AnyDType, A1, A2, A5, A4]: ...
  # END: The `@` operator


class Array5(Generic[DT, A1, A2, A3, A4, A5]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]
  T: Array5[DT, A5, A4, A3, A2, A1]
  at: Any
  ndim: Literal[5]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array5[DT, A1, A2, A3, A4, A5]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def {{ func }}(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array6(Generic[DT, A1, A2, A3, A4, A5, A6]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]
  T: Array6[DT, A6, A5, A4, A3, A2, A1]
  at: Any
  ndim: Literal[6]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array6[DT, A1, A2, A3, A4, A5, A6]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def {{ func }}(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array7(Generic[DT, A1, A2, A3, A4, A5, A6, A7]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]
  T: Array7[DT, A7, A6, A5, A4, A3, A2, A1]
  at: Any
  ndim: Literal[7]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def {{ func }}(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  {% endfor %}

  # END: Binary element-wise operators


class Array8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int]
  T: Array8[DT, A8, A7, A6, A5, A4, A3, A2, A1]
  at: Any
  ndim: Literal[8]
  dtype: type
  def __len__(self) -> int: ...
  def astype(self, dtype) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  # BEGIN: Unary operators
  {% for func in unary_funcs %}
  def {{ func }}(self) -> Array8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  {% endfor %}
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  {% for func in binary_elementwise_funcs %}

  {# Broadcasting case 1: Broadcasting with scalars #}
  @overload
  def {{ func }}(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array0[AnyDType]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# Broadcasting case 2: Broadcasting with a lesser rank #}
  @overload
  def {{ func }}(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def {{ func }}(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {# No broadcast #}
  @overload
  def {{ func }}(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  {% endfor %}

  # END: Binary element-wise operators


Array0AnyDType = Array0[AnyDType]
Array1AnyDType = Array1[AnyDType, A1]
Array2AnyDType = Array2[AnyDType, A1, A2]
Array3AnyDType = Array3[AnyDType, A1, A2, A3]
Array4AnyDType = Array4[AnyDType, A1, A2, A3, A4]
Array5AnyDType = Array5[AnyDType, A1, A2, A3, A4, A5]
Array6AnyDType = Array6[AnyDType, A1, A2, A3, A4, A5, A6]
Array7AnyDType = Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]
Array8AnyDType = Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]
# LINT.ThenChange(../jax.pyi)

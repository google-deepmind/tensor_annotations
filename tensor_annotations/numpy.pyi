"""Type stubs for custom NumPy tensor classes.

NOTE: This file is generated from templates/numpy_tensors.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_tensor_template.py \
     --template templates/numpy_tensors.pyi \
     --out numpy.pyi
"""

from typing import Any, Literal, Tuple, TypeVar, Union, Generic
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

Number = Union[
    int, float,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64
]






# A scalar array, constructed by doing eg `np.zeros(())`.
class Array0(Generic[DT]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...

  # Technically, arrays of any rank can float()ed if they only contain a
  # single value, but we can only guarantee it for Array0.
  def __float__(self) -> float: ...

  shape: Tuple[()]
  T: Array0[DT]
  ndim: Literal[0]
  dtype: type
  def astype(self, dtype) -> Array0[AnyDType]: ...

  # Technically this exists on all instances of JAX arrays, but it throws an
  # error when called on anything but scalar arrays.
  def item(self) -> Union[int, float, bool, complex]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array0[DT]: ...
  
  def __neg__(self) -> Array0[DT]: ...
  
  def __pos__(self) -> Array0[DT]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators
  
  @overload
  def __add__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __add__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __add__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __add__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __sub__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __sub__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __sub__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __sub__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __floordiv__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __floordiv__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __truediv__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __truediv__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __truediv__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __truediv__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __pow__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __pow__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __pow__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __pow__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __lt__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __lt__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __lt__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __lt__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __le__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __le__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __le__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __le__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __ge__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __ge__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __ge__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __ge__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __gt__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __gt__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __gt__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __gt__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __eq__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __eq__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __eq__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __eq__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __ne__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __ne__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __ne__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __ne__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __mul__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __mul__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __mul__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __mul__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  @overload
  def __rmul__(self, other: Array0[AnyDType]) -> Array0[AnyDType]: ...
  @overload
  def __rmul__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...
  @overload
  def __rmul__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __rmul__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  
  # END: Binary element-wise operators


class Array1(Generic[DT, A1]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int]
  T: Array1[DT, A1]
  ndim: Literal[1]
  dtype: type
  def astype(self, dtype) -> Array1[AnyDType, A1]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array1[DT, A1]: ...
  
  def __neg__(self) -> Array1[DT, A1]: ...
  
  def __pos__(self) -> Array1[DT, A1]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array1[AnyDType, A1]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A1]) -> Array1[AnyDType, A1]: ...

  

  # END: Binary element-wise operators

  # BEGIN: The `@` operator
  @overload
  def __matmul__(self, other: Array1[AnyDType, A1]) -> AnyDType: ...

  @overload
  def __matmul__(self, other: Array2[AnyDType, A1, A2]) -> Array1[AnyDType, A2]: ...

  @overload
  def __matmul__(self, other: Array3[AnyDType, A3, A1, A2]) -> Array2[AnyDType, A3, A2]: ...

  @overload
  def __matmul__(self, other: Array4[AnyDType, A3, A4, A1, A2]) -> Array3[AnyDType, A3, A4, A2]: ...

  @overload
  def __rmatmul__(self, other: Array1[AnyDType, A1]) -> AnyDType: ...

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
  ndim: Literal[2]
  dtype: type
  def astype(self, dtype) -> Array2[AnyDType, A1, A2]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array2[DT, A1, A2]: ...
  
  def __neg__(self) -> Array2[DT, A1, A2]: ...
  
  def __pos__(self) -> Array2[DT, A1, A2]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __add__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __sub__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __truediv__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __pow__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __lt__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __le__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __ge__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __gt__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __eq__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __ne__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __mul__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A2]) -> Array2[AnyDType, A1, A2]: ...

  
  @overload
  def __rmul__(self, other: Array2[AnyDType, A1, A2]) -> Array2[AnyDType, A1, A2]: ...

  

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
  ndim: Literal[3]
  dtype: type
  def astype(self, dtype) -> Array3[AnyDType, A1, A2, A3]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array3[DT, A1, A2, A3]: ...
  
  def __neg__(self) -> Array3[DT, A1, A2, A3]: ...
  
  def __pos__(self) -> Array3[DT, A1, A2, A3]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __add__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __sub__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __truediv__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __pow__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __lt__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __le__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __ge__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __gt__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __eq__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __ne__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __mul__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A3]) -> Array3[AnyDType, A1, A2, A3]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  
  @overload
  def __rmul__(self, other: Array3[AnyDType, A1, A2, A3]) -> Array3[AnyDType, A1, A2, A3]: ...

  

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
  ndim: Literal[4]
  dtype: type
  def astype(self, dtype) -> Array4[AnyDType]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array4[DT, A1, A2, A3, A4]: ...
  
  def __neg__(self) -> Array4[DT, A1, A2, A3, A4]: ...
  
  def __pos__(self) -> Array4[DT, A1, A2, A3, A4]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __add__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __add__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __sub__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __sub__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __floordiv__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __truediv__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __truediv__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __pow__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __pow__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __lt__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __lt__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __le__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __le__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __ge__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __ge__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __gt__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __gt__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __eq__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __eq__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __ne__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __ne__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __mul__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __mul__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...
  @overload
  def __rmul__(self, other: Array3[AnyDType, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  
  @overload
  def __rmul__(self, other: Array4[AnyDType, A1, A2, A3, A4]) -> Array4[AnyDType, A1, A2, A3, A4]: ...

  

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
  shape: Tuple[int, int, int, int, int]
  T: Array5[DT, A5, A4, A3, A2, A1]
  ndim: Literal[5]
  dtype: type
  def astype(self, dtype) -> Array5[AnyDType]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array5[DT, A1, A2, A3, A4, A5]: ...
  
  def __neg__(self) -> Array5[DT, A1, A2, A3, A4, A5]: ...
  
  def __pos__(self) -> Array5[DT, A1, A2, A3, A4, A5]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __add__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __add__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __add__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __sub__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __sub__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __sub__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __floordiv__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __floordiv__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __truediv__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __truediv__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __truediv__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __pow__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __pow__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __pow__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __lt__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __lt__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __lt__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __le__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __le__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __le__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __ge__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __ge__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __ge__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __gt__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __gt__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __gt__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __eq__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __eq__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __eq__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __ne__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __ne__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __ne__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __mul__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __mul__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __mul__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __rmul__(self, other: Array3[AnyDType, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...
  @overload
  def __rmul__(self, other: Array4[AnyDType, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  
  @overload
  def __rmul__(self, other: Array5[AnyDType, A1, A2, A3, A4, A5]) -> Array5[AnyDType, A1, A2, A3, A4, A5]: ...

  

  # END: Binary element-wise operators


class Array6(Generic[DT, A1, A2, A3, A4, A5, A6]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int, int, int]
  T: Array6[DT, A6, A5, A4, A3, A2, A1]
  ndim: Literal[6]
  dtype: type
  def astype(self, dtype) -> Array6[AnyDType]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array6[DT, A1, A2, A3, A4, A5, A6]: ...
  
  def __neg__(self) -> Array6[DT, A1, A2, A3, A4, A5, A6]: ...
  
  def __pos__(self) -> Array6[DT, A1, A2, A3, A4, A5, A6]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __add__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __add__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __add__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __add__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __sub__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __sub__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __sub__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __sub__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __floordiv__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __floordiv__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __floordiv__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __truediv__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __truediv__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __truediv__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __truediv__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __pow__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __pow__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __pow__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __pow__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __lt__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __lt__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __lt__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __lt__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __le__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __le__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __le__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __le__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ge__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ge__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ge__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __ge__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __gt__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __gt__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __gt__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __gt__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __eq__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __eq__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __eq__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __eq__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ne__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ne__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __ne__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __ne__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __mul__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __mul__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __mul__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __mul__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __rmul__(self, other: Array3[AnyDType, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __rmul__(self, other: Array4[AnyDType, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...
  @overload
  def __rmul__(self, other: Array5[AnyDType, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  
  @overload
  def __rmul__(self, other: Array6[AnyDType, A1, A2, A3, A4, A5, A6]) -> Array6[AnyDType, A1, A2, A3, A4, A5, A6]: ...

  

  # END: Binary element-wise operators


class Array7(Generic[DT, A1, A2, A3, A4, A5, A6, A7]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int, int, int, int]
  T: Array7[DT, A7, A6, A5, A4, A3, A2, A1]
  ndim: Literal[7]
  dtype: type
  def astype(self, dtype) -> Array7[AnyDType]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  
  def __neg__(self) -> Array7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  
  def __pos__(self) -> Array7[DT, A1, A2, A3, A4, A5, A6, A7]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __add__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __add__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __add__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __add__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __add__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __sub__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __sub__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __sub__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __sub__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __sub__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __floordiv__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __floordiv__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __floordiv__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __floordiv__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __truediv__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __truediv__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __truediv__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __truediv__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __truediv__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __pow__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __pow__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __pow__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __pow__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __pow__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __lt__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __lt__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __lt__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __lt__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __lt__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __le__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __le__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __le__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __le__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __le__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ge__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ge__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ge__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ge__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __ge__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __gt__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __gt__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __gt__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __gt__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __gt__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __eq__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __eq__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __eq__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __eq__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __eq__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ne__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ne__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ne__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __ne__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __ne__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __mul__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __mul__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __mul__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __mul__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __mul__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __rmul__(self, other: Array3[AnyDType, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __rmul__(self, other: Array4[AnyDType, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __rmul__(self, other: Array5[AnyDType, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...
  @overload
  def __rmul__(self, other: Array6[AnyDType, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  
  @overload
  def __rmul__(self, other: Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]) -> Array7[AnyDType, A1, A2, A3, A4, A5, A6, A7]: ...

  

  # END: Binary element-wise operators


class Array8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8]):
  def __getitem__(self, index) -> Any: ...
  def __setitem__(self, index, value) -> Any: ...
  shape: Tuple[int, int, int, int, int, int, int, int]
  T: Array8[DT, A8, A7, A6, A5, A4, A3, A2, A1]
  ndim: Literal[8]
  dtype: type
  def astype(self, dtype) -> Array8[AnyDType]: ...

  # BEGIN: Unary operators
  
  def __abs__(self) -> Array8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  
  def __neg__(self) -> Array8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  
  def __pos__(self) -> Array8[DT, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  
  # END: Unary operators

  # BEGIN: Binary element-wise operators

  

  
  @overload
  def __add__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __add__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __add__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __add__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __add__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __add__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __add__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __add__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __add__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __sub__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __sub__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __sub__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __sub__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __sub__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __sub__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __sub__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __sub__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __sub__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __floordiv__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __floordiv__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __floordiv__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __floordiv__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __floordiv__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __floordiv__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __floordiv__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __floordiv__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __floordiv__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __truediv__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __truediv__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __truediv__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __truediv__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __truediv__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __truediv__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __truediv__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __truediv__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __truediv__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __pow__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __pow__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __pow__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __pow__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __pow__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __pow__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __pow__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __pow__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __pow__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __lt__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __lt__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __lt__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __lt__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __lt__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __lt__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __lt__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __lt__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __lt__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __le__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __le__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __le__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __le__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __le__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __le__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __le__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __le__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __le__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __ge__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __ge__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ge__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ge__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ge__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ge__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ge__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ge__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __ge__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __gt__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __gt__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __gt__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __gt__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __gt__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __gt__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __gt__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __gt__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __gt__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __eq__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __eq__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __eq__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __eq__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __eq__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __eq__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __eq__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __eq__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __eq__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __ne__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __ne__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ne__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ne__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ne__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ne__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ne__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __ne__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __ne__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __mul__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __mul__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __mul__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __mul__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __mul__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __mul__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __mul__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __mul__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __mul__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  
  @overload
  def __rmul__(self, other: Number) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __rmul__(self, other: Array1[AnyDType, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __rmul__(self, other: Array2[AnyDType, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __rmul__(self, other: Array3[AnyDType, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __rmul__(self, other: Array4[AnyDType, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __rmul__(self, other: Array5[AnyDType, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __rmul__(self, other: Array6[AnyDType, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...
  @overload
  def __rmul__(self, other: Array7[AnyDType, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  
  @overload
  def __rmul__(self, other: Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]) -> Array8[AnyDType, A1, A2, A3, A4, A5, A6, A7, A8]: ...

  

  # END: Binary element-wise operators


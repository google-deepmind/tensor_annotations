# LINT.IfChange
"""Type stubs for custom NumPy tensor classes.

NOTE: This file is generated from templates/numpy_tensors.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_tensor_template.py \
     --template templates/numpy_tensors.pyi \
     --out numpy.pyi
"""

from typing import Any, TypeVar, Generic
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


class Array1(Generic[DT, A1]):
  pass


class Array2(Generic[DT, A1, A2]):
  pass


class Array3(Generic[DT, A1, A2, A3]):
  pass


class Array4(Generic[DT, A1, A2, A3, A4]):
  pass


class Array5(Generic[DT, A1, A2, A3, A4, A5]):
  pass


class Array6(Generic[DT, A1, A2, A3, A4, A5, A6]):
  pass


class Array7(Generic[DT, A1, A2, A3, A4, A5, A6, A7]):
  pass


class Array8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8]):
  pass

# LINT.ThenChange(../numpy.pyi)

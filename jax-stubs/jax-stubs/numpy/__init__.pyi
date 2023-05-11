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
"""Stubs for jax.numpy.*

We also need to provide stubs for jax.Array in here to avoid breaking
code which doesn't use tensor_annotations annotations. Le sigh.

NOTE: This file is generated from templates/jax_numpy.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_jax_library_template.py
"""

from typing import overload, Any, List, Literal, Tuple, TypeVar

import jax
import tensor_annotations.jax as tjax
from tensor_annotations.jax import Array0, Array1, Array2, Array3, Array4
from tensor_annotations.axes import Axis


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)

AnyDType = Any
DT = TypeVar('DT', bound=tjax.DType)

Shape0 = Tuple[()]
Shape1 = Tuple[int]
Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Shape4 = Tuple[int, int, int, int]

L0 = Literal[0]
L1 = Literal[1]
L2 = Literal[2]
L3 = Literal[3]
L4 = Literal[4]
LN1 = Literal[-1]

# Users might have some variables explicitly annotated as ndarray, so we need
# to provide minimal stubs for ndarray too so users don't get type errors.
class ndarray:

  def __getitem__(self, key) -> ndarray:
    pass

# ---------- UNARY OPERATORS ----------





@overload
def abs(x: jax.Array) -> jax.Array: ...


@overload
def abs(x: Array0[DT]) -> Array0[DT]: ...


@overload
def abs(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def abs(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def abs(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def abs(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def acos(x: jax.Array) -> jax.Array: ...


@overload
def acos(x: Array0[DT]) -> Array0[DT]: ...


@overload
def acos(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def acos(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def acos(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def acos(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def acosh(x: jax.Array) -> jax.Array: ...


@overload
def acosh(x: Array0[DT]) -> Array0[DT]: ...


@overload
def acosh(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def acosh(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def acosh(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def acosh(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def asin(x: jax.Array) -> jax.Array: ...


@overload
def asin(x: Array0[DT]) -> Array0[DT]: ...


@overload
def asin(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def asin(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def asin(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def asin(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def asinh(x: jax.Array) -> jax.Array: ...


@overload
def asinh(x: Array0[DT]) -> Array0[DT]: ...


@overload
def asinh(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def asinh(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def asinh(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def asinh(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def atan(x: jax.Array) -> jax.Array: ...


@overload
def atan(x: Array0[DT]) -> Array0[DT]: ...


@overload
def atan(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def atan(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def atan(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def atan(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def atanh(x: jax.Array) -> jax.Array: ...


@overload
def atanh(x: Array0[DT]) -> Array0[DT]: ...


@overload
def atanh(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def atanh(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def atanh(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def atanh(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def cos(x: jax.Array) -> jax.Array: ...


@overload
def cos(x: Array0[DT]) -> Array0[DT]: ...


@overload
def cos(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def cos(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def cos(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def cos(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def cosh(x: jax.Array) -> jax.Array: ...


@overload
def cosh(x: Array0[DT]) -> Array0[DT]: ...


@overload
def cosh(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def cosh(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def cosh(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def cosh(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def exp(x: jax.Array) -> jax.Array: ...


@overload
def exp(x: Array0[DT]) -> Array0[DT]: ...


@overload
def exp(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def exp(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def exp(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def exp(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def floor(x: jax.Array) -> jax.Array: ...


@overload
def floor(x: Array0[DT]) -> Array0[DT]: ...


@overload
def floor(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def floor(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def floor(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def floor(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def logical_not(x: jax.Array) -> jax.Array: ...


@overload
def logical_not(x: Array0[DT]) -> Array0[DT]: ...


@overload
def logical_not(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def logical_not(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def logical_not(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def logical_not(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def negative(x: jax.Array) -> jax.Array: ...


@overload
def negative(x: Array0[DT]) -> Array0[DT]: ...


@overload
def negative(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def negative(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def negative(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def negative(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def sigmoid(x: jax.Array) -> jax.Array: ...


@overload
def sigmoid(x: Array0[DT]) -> Array0[DT]: ...


@overload
def sigmoid(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def sigmoid(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def sigmoid(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def sigmoid(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def sign(x: jax.Array) -> jax.Array: ...


@overload
def sign(x: Array0[DT]) -> Array0[DT]: ...


@overload
def sign(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def sign(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def sign(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def sign(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def sin(x: jax.Array) -> jax.Array: ...


@overload
def sin(x: Array0[DT]) -> Array0[DT]: ...


@overload
def sin(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def sin(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def sin(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def sin(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def sinh(x: jax.Array) -> jax.Array: ...


@overload
def sinh(x: Array0[DT]) -> Array0[DT]: ...


@overload
def sinh(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def sinh(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def sinh(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def sinh(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def sqrt(x: jax.Array) -> jax.Array: ...


@overload
def sqrt(x: Array0[DT]) -> Array0[DT]: ...


@overload
def sqrt(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def sqrt(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def sqrt(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def sqrt(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def square(x: jax.Array) -> jax.Array: ...


@overload
def square(x: Array0[DT]) -> Array0[DT]: ...


@overload
def square(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def square(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def square(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def square(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def tan(x: jax.Array) -> jax.Array: ...


@overload
def tan(x: Array0[DT]) -> Array0[DT]: ...


@overload
def tan(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def tan(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def tan(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def tan(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...




@overload
def tanh(x: jax.Array) -> jax.Array: ...


@overload
def tanh(x: Array0[DT]) -> Array0[DT]: ...


@overload
def tanh(x: Array1[DT, A1]) -> Array1[DT, A1]: ...


@overload
def tanh(x: Array2[DT, A1, A2]) -> Array2[DT, A1, A2]: ...


@overload
def tanh(x: Array3[DT, A1, A2, A3]) -> Array3[DT, A1, A2, A3]: ...


@overload
def tanh(x: Array4[DT, A1, A2, A3, A4]) -> Array4[DT, A1, A2, A3, A4]: ...







@overload
def zeros_like(x: jax.Array, dtype=...) -> jax.Array: ...

@overload
def zeros_like(x: Array0[DT], dtype=...) -> Array0[DT]: ...


@overload
def zeros_like(x: Array1[DT, A1], dtype=...) -> Array1[DT, A1]: ...


@overload
def zeros_like(x: Array2[DT, A1, A2], dtype=...) -> Array2[DT, A1, A2]: ...


@overload
def zeros_like(x: Array3[DT, A1, A2, A3], dtype=...) -> Array3[DT, A1, A2, A3]: ...


@overload
def zeros_like(x: Array4[DT, A1, A2, A3, A4], dtype=...) -> Array4[DT, A1, A2, A3, A4]: ...



@overload
def ones_like(x: jax.Array, dtype=...) -> jax.Array: ...

@overload
def ones_like(x: Array0[DT], dtype=...) -> Array0[DT]: ...


@overload
def ones_like(x: Array1[DT, A1], dtype=...) -> Array1[DT, A1]: ...


@overload
def ones_like(x: Array2[DT, A1, A2], dtype=...) -> Array2[DT, A1, A2]: ...


@overload
def ones_like(x: Array3[DT, A1, A2, A3], dtype=...) -> Array3[DT, A1, A2, A3]: ...


@overload
def ones_like(x: Array4[DT, A1, A2, A3, A4], dtype=...) -> Array4[DT, A1, A2, A3, A4]: ...



@overload
def round(x: jax.Array, decimals=...) -> jax.Array: ...

@overload
def round(x: Array0[DT], decimals=...) -> Array0[DT]: ...


@overload
def round(x: Array1[DT, A1], decimals=...) -> Array1[DT, A1]: ...


@overload
def round(x: Array2[DT, A1, A2], decimals=...) -> Array2[DT, A1, A2]: ...


@overload
def round(x: Array3[DT, A1, A2, A3], decimals=...) -> Array3[DT, A1, A2, A3]: ...


@overload
def round(x: Array4[DT, A1, A2, A3, A4], decimals=...) -> Array4[DT, A1, A2, A3, A4]: ...


# I what even why would you
@overload
def sqrt(x: float) -> float: ...

# ---------- ZEROS, ONES ----------

# Can't type these properly when shape is specified as a list. :(
# But if shape is specified as an int or a tuple, we're good! :)

@overload
def zeros(shape: List, dtype=...) -> Any: ...

@overload
def zeros(shape: int) -> Array1[tjax.float32, Any]: ...

@overload
def zeros(shape: int, dtype=...) -> Array1[AnyDType, Any]: ...





@overload
def zeros(shape: Shape0) -> Array0[tjax.float32, ]: ...

@overload
def zeros(shape: Shape0, dtype=...) -> Array0[AnyDType, ]: ...




@overload
def zeros(shape: Shape1) -> Array1[tjax.float32, Any]: ...

@overload
def zeros(shape: Shape1, dtype=...) -> Array1[AnyDType, Any]: ...




@overload
def zeros(shape: Shape2) -> Array2[tjax.float32, Any, Any]: ...

@overload
def zeros(shape: Shape2, dtype=...) -> Array2[AnyDType, Any, Any]: ...




@overload
def zeros(shape: Shape3) -> Array3[tjax.float32, Any, Any, Any]: ...

@overload
def zeros(shape: Shape3, dtype=...) -> Array3[AnyDType, Any, Any, Any]: ...




@overload
def zeros(shape: Shape4) -> Array4[tjax.float32, Any, Any, Any, Any]: ...

@overload
def zeros(shape: Shape4, dtype=...) -> Array4[AnyDType, Any, Any, Any, Any]: ...




@overload
def ones(shape: List, dtype=...) -> Any: ...

@overload
def ones(shape: int) -> Array1[tjax.float32, Any]: ...

@overload
def ones(shape: int, dtype=...) -> Array1[AnyDType, Any]: ...





@overload
def ones(shape: Shape0) -> Array0[tjax.float32, ]: ...

@overload
def ones(shape: Shape0, dtype=...) -> Array0[AnyDType, ]: ...




@overload
def ones(shape: Shape1) -> Array1[tjax.float32, Any]: ...

@overload
def ones(shape: Shape1, dtype=...) -> Array1[AnyDType, Any]: ...




@overload
def ones(shape: Shape2) -> Array2[tjax.float32, Any, Any]: ...

@overload
def ones(shape: Shape2, dtype=...) -> Array2[AnyDType, Any, Any]: ...




@overload
def ones(shape: Shape3) -> Array3[tjax.float32, Any, Any, Any]: ...

@overload
def ones(shape: Shape3, dtype=...) -> Array3[AnyDType, Any, Any, Any]: ...




@overload
def ones(shape: Shape4) -> Array4[tjax.float32, Any, Any, Any, Any]: ...

@overload
def ones(shape: Shape4, dtype=...) -> Array4[AnyDType, Any, Any, Any, Any]: ...





# ---------- REDUCTION OPERATORS ----------



@overload
def sum(
    a: jax.Array,
    keepdims=...,
    axis=...,
    out=...,
    dtype=...
) -> jax.Array: ...

## keepdims = True: yet be to be typed

# This type signature is technically incorrect: `keepdims` is *not*
# the second argument. However, this way seems to be the only way
# to get both pytype and Mypy to recognise this overload: if we put
# `keepdims: Literal[True]` in the right position, Mypy complains
# about a non-default argument following a default argument; if we
# put `keepdims: Literal[True] = ...` in the right place, pytype
# matches this overload even when `keepdims=False`.
#
# In practice, though, this shouldn't be an issue:
# * It's very unlikely that anyone would pass `True` as the second (non-keyword)
#   argument here (since the second argument is *supposed* to be `axis`).
# * If someone *did* want to set `keepdims` to `True`, they'd *have* to
#   use a keyword argument, since `keepdims` comes after `out, and setting `out`
#   to anything (even `None`) produces a "The 'out' argument to jnp.sum is
#   not supported" error.
@overload
def sum(
    a: Any,
    keepdims: Literal[True],
    axis=...,
    out=...,
    dtype=...
) -> Any: ...

## keepdims = False or unspecified



### n_axes = 1

#### `axis` specified



@overload
def sum(
    a: Array1[DT, A1],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def sum(
    a: Array1[DT, A1],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def sum(
    a: Array1[DT, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def sum(
    a: Array1[DT, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### n_axes = 2

#### `axis` specified



@overload
def sum(
    a: Array2[DT, A1, A2],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def sum(
    a: Array2[DT, A1, A2],
    axis: L1,
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def sum(
    a: Array2[DT, A1, A2],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def sum(
    a: Array2[DT, A1, A2],
    axis: Tuple[L0, L1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def sum(
    a: Array2[DT, A1, A2],
    axis: Tuple[L0, LN1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def sum(
    a: Array2[DT, Any, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def sum(
    a: Array2[DT, Any, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### n_axes = 3

#### `axis` specified



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A3]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: L1,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A3]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: L2,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A3]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L2],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def sum(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def sum(
    a: Array3[DT, Any, Any, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def sum(
    a: Array3[DT, Any, Any, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### n_axes = 4

#### `axis` specified



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A2, A3, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L1,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A3, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L2,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A2, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L3,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A2, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A2, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A3, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L2],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L3],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, LN1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L3],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A4]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L3],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A3]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def sum(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def sum(
    a: Array4[DT, Any, Any, Any, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def sum(
    a: Array4[DT, Any, Any, Any, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### Some weird argument like a list of arrays

@overload
def sum(
    a: Any,
    axis=...,
    out=..., keepdims=..., dtype=...
) -> Any: ...



@overload
def mean(
    a: jax.Array,
    keepdims=...,
    axis=...,
    out=...,
    dtype=...
) -> jax.Array: ...

## keepdims = True: yet be to be typed

# This type signature is technically incorrect: `keepdims` is *not*
# the second argument. However, this way seems to be the only way
# to get both pytype and Mypy to recognise this overload: if we put
# `keepdims: Literal[True]` in the right position, Mypy complains
# about a non-default argument following a default argument; if we
# put `keepdims: Literal[True] = ...` in the right place, pytype
# matches this overload even when `keepdims=False`.
#
# In practice, though, this shouldn't be an issue:
# * It's very unlikely that anyone would pass `True` as the second (non-keyword)
#   argument here (since the second argument is *supposed* to be `axis`).
# * If someone *did* want to set `keepdims` to `True`, they'd *have* to
#   use a keyword argument, since `keepdims` comes after `out, and setting `out`
#   to anything (even `None`) produces a "The 'out' argument to jnp.mean is
#   not supported" error.
@overload
def mean(
    a: Any,
    keepdims: Literal[True],
    axis=...,
    out=...,
    dtype=...
) -> Any: ...

## keepdims = False or unspecified



### n_axes = 1

#### `axis` specified



@overload
def mean(
    a: Array1[DT, A1],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def mean(
    a: Array1[DT, A1],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def mean(
    a: Array1[DT, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def mean(
    a: Array1[DT, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### n_axes = 2

#### `axis` specified



@overload
def mean(
    a: Array2[DT, A1, A2],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def mean(
    a: Array2[DT, A1, A2],
    axis: L1,
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def mean(
    a: Array2[DT, A1, A2],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def mean(
    a: Array2[DT, A1, A2],
    axis: Tuple[L0, L1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def mean(
    a: Array2[DT, A1, A2],
    axis: Tuple[L0, LN1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def mean(
    a: Array2[DT, Any, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def mean(
    a: Array2[DT, Any, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### n_axes = 3

#### `axis` specified



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A3]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: L1,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A3]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: L2,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A3]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L2],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def mean(
    a: Array3[DT, A1, A2, A3],
    axis: Tuple[L0, L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def mean(
    a: Array3[DT, Any, Any, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def mean(
    a: Array3[DT, Any, Any, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### n_axes = 4

#### `axis` specified



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L0,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A2, A3, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L1,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A3, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L2,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A2, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: L3,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A2, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: LN1,
    out=..., keepdims=..., dtype=...
) -> Array3[DT, A1, A2, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A3, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L2],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L3],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, LN1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A2, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L3],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array2[DT, A1, A2]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L2],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A4]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L3],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A3]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A2]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L1, L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array1[DT, A1]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L2, L3],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



@overload
def mean(
    a: Array4[DT, A1, A2, A3, A4],
    axis: Tuple[L0, L1, L2, LN1],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



# Fallback: `axis` not any of the above
@overload
def mean(
    a: Array4[DT, Any, Any, Any, Any],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def mean(
    a: Array4[DT, Any, Any, Any, Any],
    out=..., keepdims=..., dtype=...
) -> Array0[DT]: ...



### Some weird argument like a list of arrays

@overload
def mean(
    a: Any,
    axis=...,
    out=..., keepdims=..., dtype=...
) -> Any: ...



# ---------- TRANSPOSE ----------

@overload
def transpose(
    a: jax.Array,
    axes=...
) -> jax.Array: ...



### n_axes = 1

#### `axes` specified



@overload
def transpose(
    a: Array1[DT, A1],
    axes: Tuple[L0]
) -> Array1[DT, A1]: ...



#### `axes` unspecified






@overload
def transpose(
    a: Array1[DT, A1]
) -> Array1[DT, A1]: ...



### n_axes = 2

#### `axes` specified



@overload
def transpose(
    a: Array2[DT, A1, A2],
    axes: Tuple[L0, L1]
) -> Array2[DT, A1, A2]: ...



@overload
def transpose(
    a: Array2[DT, A1, A2],
    axes: Tuple[L1, L0]
) -> Array2[DT, A2, A1]: ...



#### `axes` unspecified






@overload
def transpose(
    a: Array2[DT, A1, A2]
) -> Array2[DT, A2, A1]: ...



### n_axes = 3

#### `axes` specified



@overload
def transpose(
    a: Array3[DT, A1, A2, A3],
    axes: Tuple[L0, L1, L2]
) -> Array3[DT, A1, A2, A3]: ...



@overload
def transpose(
    a: Array3[DT, A1, A2, A3],
    axes: Tuple[L0, L2, L1]
) -> Array3[DT, A1, A3, A2]: ...



@overload
def transpose(
    a: Array3[DT, A1, A2, A3],
    axes: Tuple[L1, L0, L2]
) -> Array3[DT, A2, A1, A3]: ...



@overload
def transpose(
    a: Array3[DT, A1, A2, A3],
    axes: Tuple[L1, L2, L0]
) -> Array3[DT, A2, A3, A1]: ...



@overload
def transpose(
    a: Array3[DT, A1, A2, A3],
    axes: Tuple[L2, L0, L1]
) -> Array3[DT, A3, A1, A2]: ...



@overload
def transpose(
    a: Array3[DT, A1, A2, A3],
    axes: Tuple[L2, L1, L0]
) -> Array3[DT, A3, A2, A1]: ...



#### `axes` unspecified






@overload
def transpose(
    a: Array3[DT, A1, A2, A3]
) -> Array3[DT, A3, A2, A1]: ...



### n_axes = 4

#### `axes` specified



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L0, L1, L2, L3]
) -> Array4[DT, A1, A2, A3, A4]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L0, L1, L3, L2]
) -> Array4[DT, A1, A2, A4, A3]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L0, L2, L1, L3]
) -> Array4[DT, A1, A3, A2, A4]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L0, L2, L3, L1]
) -> Array4[DT, A1, A3, A4, A2]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L0, L3, L1, L2]
) -> Array4[DT, A1, A4, A2, A3]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L0, L3, L2, L1]
) -> Array4[DT, A1, A4, A3, A2]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L1, L0, L2, L3]
) -> Array4[DT, A2, A1, A3, A4]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L1, L0, L3, L2]
) -> Array4[DT, A2, A1, A4, A3]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L1, L2, L0, L3]
) -> Array4[DT, A2, A3, A1, A4]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L1, L2, L3, L0]
) -> Array4[DT, A2, A3, A4, A1]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L1, L3, L0, L2]
) -> Array4[DT, A2, A4, A1, A3]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L1, L3, L2, L0]
) -> Array4[DT, A2, A4, A3, A1]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L2, L0, L1, L3]
) -> Array4[DT, A3, A1, A2, A4]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L2, L0, L3, L1]
) -> Array4[DT, A3, A1, A4, A2]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L2, L1, L0, L3]
) -> Array4[DT, A3, A2, A1, A4]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L2, L1, L3, L0]
) -> Array4[DT, A3, A2, A4, A1]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L2, L3, L0, L1]
) -> Array4[DT, A3, A4, A1, A2]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L2, L3, L1, L0]
) -> Array4[DT, A3, A4, A2, A1]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L3, L0, L1, L2]
) -> Array4[DT, A4, A1, A2, A3]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L3, L0, L2, L1]
) -> Array4[DT, A4, A1, A3, A2]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L3, L1, L0, L2]
) -> Array4[DT, A4, A2, A1, A3]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L3, L1, L2, L0]
) -> Array4[DT, A4, A2, A3, A1]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L3, L2, L0, L1]
) -> Array4[DT, A4, A3, A1, A2]: ...



@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4],
    axes: Tuple[L3, L2, L1, L0]
) -> Array4[DT, A4, A3, A2, A1]: ...



#### `axes` unspecified






@overload
def transpose(
    a: Array4[DT, A1, A2, A3, A4]
) -> Array4[DT, A4, A3, A2, A1]: ...



# ---------- EVERYTHING ELSE: UNTYPED ----------


ComplexWarning: Any

NINF: Any

NZERO: Any

PZERO: Any

absolute: Any

add: Any

all: Any

allclose: Any

alltrue: Any

amax: Any

amin: Any

angle: Any

any: Any

append: Any

apply_along_axis: Any

apply_over_axes: Any

arange: Any

arccos: Any

arccosh: Any

arcsin: Any

arcsinh: Any

arctan: Any

arctan2: Any

arctanh: Any

argmax: Any

argmin: Any

argpartition: Any

argsort: Any

argwhere: Any

around: Any

array: Any

array_equal: Any

array_equiv: Any

array_repr: Any

array_split: Any

array_str: Any

asarray: Any

atleast_1d: Any

atleast_2d: Any

atleast_3d: Any

average: Any

bartlett: Any

bfloat16: Any

bincount: Any

bitwise_and: Any

bitwise_not: Any

bitwise_or: Any

bitwise_xor: Any

blackman: Any

block: Any

bool_: Any

broadcast_arrays: Any

broadcast_shapes: Any

broadcast_to: Any

c_: Any

can_cast: Any

cbrt: Any

cdouble: Any

ceil: Any

character: Any

choose: Any

clip: Any

column_stack: Any

complex128: Any

complex64: Any

complex_: Any

complexfloating: Any

compress: Any

concatenate: Any

conj: Any

conjugate: Any

convolve: Any

copy: Any

copysign: Any

corrcoef: Any

correlate: Any

count_nonzero: Any

cov: Any

cross: Any

csingle: Any

cumprod: Any

cumproduct: Any

cumsum: Any

deg2rad: Any

degrees: Any

delete: Any

diag: Any

diag_indices: Any

diag_indices_from: Any

diagflat: Any

diagonal: Any

diff: Any

digitize: Any

divide: Any

divmod: Any

dot: Any

double: Any

dsplit: Any

dstack: Any

dtype: Any

e: Any

ediff1d: Any

einsum: Any

einsum_path: Any

empty: Any

empty_like: Any

equal: Any

euler_gamma: Any

exp2: Any

expand_dims: Any

expm1: Any

extract: Any

eye: Any

fabs: Any

fft: Any

finfo: Any

fix: Any

flatnonzero: Any

flexible: Any

flip: Any

fliplr: Any

flipud: Any

float16: Any

float32: Any

float64: Any

float8_e4m3fn: Any

float8_e5m2: Any

float_: Any

float_power: Any

floating: Any

floor_divide: Any

fmax: Any

fmin: Any

fmod: Any

frexp: Any

from_dlpack: Any

frombuffer: Any

fromfile: Any

fromfunction: Any

fromiter: Any

fromstring: Any

full: Any

full_like: Any

gcd: Any

generic: Any

geomspace: Any

get_printoptions: Any

gradient: Any

greater: Any

greater_equal: Any

hamming: Any

hanning: Any

heaviside: Any

histogram: Any

histogram2d: Any

histogram_bin_edges: Any

histogramdd: Any

hsplit: Any

hstack: Any

hypot: Any

i0: Any

identity: Any

iinfo: Any

imag: Any

in1d: Any

index_exp: Any

indices: Any

inexact: Any

inf: Any

inner: Any

insert: Any

int16: Any

int32: Any

int64: Any

int8: Any

int_: Any

integer: Any

interp: Any

intersect1d: Any

invert: Any

isclose: Any

iscomplex: Any

iscomplexobj: Any

isfinite: Any

isin: Any

isinf: Any

isnan: Any

isneginf: Any

isposinf: Any

isreal: Any

isrealobj: Any

isscalar: Any

issubdtype: Any

issubsctype: Any

iterable: Any

ix_: Any

kaiser: Any

kron: Any

lcm: Any

ldexp: Any

left_shift: Any

less: Any

less_equal: Any

lexsort: Any

linalg: Any

linspace: Any

load: Any

log: Any

log10: Any

log1p: Any

log2: Any

logaddexp: Any

logaddexp2: Any

logical_and: Any

logical_or: Any

logical_xor: Any

logspace: Any

mask_indices: Any

matmul: Any

max: Any

maximum: Any

median: Any

meshgrid: Any

mgrid: Any

min: Any

minimum: Any

mod: Any

modf: Any

moveaxis: Any

multiply: Any

nan: Any

nan_to_num: Any

nanargmax: Any

nanargmin: Any

nancumprod: Any

nancumsum: Any

nanmax: Any

nanmean: Any

nanmedian: Any

nanmin: Any

nanpercentile: Any

nanprod: Any

nanquantile: Any

nanstd: Any

nansum: Any

nanvar: Any

ndim: Any

newaxis: Any

nextafter: Any

nonzero: Any

not_equal: Any

number: Any

object_: Any

ogrid: Any

outer: Any

packbits: Any

pad: Any

partition: Any

percentile: Any

pi: Any

piecewise: Any

place: Any

poly: Any

polyadd: Any

polyder: Any

polydiv: Any

polyfit: Any

polyint: Any

polymul: Any

polysub: Any

polyval: Any

positive: Any

power: Any

printoptions: Any

prod: Any

product: Any

promote_types: Any

ptp: Any

put: Any

quantile: Any

r_: Any

rad2deg: Any

radians: Any

ravel: Any

ravel_multi_index: Any

real: Any

reciprocal: Any

remainder: Any

repeat: Any

reshape: Any

resize: Any

result_type: Any

right_shift: Any

rint: Any

roll: Any

rollaxis: Any

roots: Any

rot90: Any

round_: Any

row_stack: Any

s_: Any

save: Any

savez: Any

searchsorted: Any

select: Any

set_printoptions: Any

setdiff1d: Any

setxor1d: Any

shape: Any

signbit: Any

signedinteger: Any

sinc: Any

single: Any

size: Any

sometrue: Any

sort: Any

sort_complex: Any

split: Any

squeeze: Any

stack: Any

std: Any

subtract: Any

swapaxes: Any

take: Any

take_along_axis: Any

tensordot: Any

tile: Any

trace: Any

trapz: Any

tri: Any

tril: Any

tril_indices: Any

tril_indices_from: Any

trim_zeros: Any

triu: Any

triu_indices: Any

triu_indices_from: Any

true_divide: Any

trunc: Any

uint: Any

uint16: Any

uint32: Any

uint64: Any

uint8: Any

union1d: Any

unique: Any

unpackbits: Any

unravel_index: Any

unsignedinteger: Any

unwrap: Any

vander: Any

var: Any

vdot: Any

vectorize: Any

vsplit: Any

vstack: Any

where: Any


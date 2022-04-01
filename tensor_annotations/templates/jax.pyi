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
# LINT.IfChange
"""JAX stubs.

NOTE: This file is generated from templates/jax.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_jax_library_template.py
"""

from typing import overload, Any, List, Literal, Tuple, TypeVar

from tensor_annotations.jax import Array0, Array1, Array2, Array3, Array4
from tensor_annotations.axes import Axis


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)

class ndarray: ...

Shape1 = Tuple[int]
Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Shape4 = Tuple[int, int, int, int]

L0 = Literal[0]
L1 = Literal[1]
L2 = Literal[2]
L3 = Literal[3]
L4 = Literal[4]


# ---------- UNARY OPERATORS ----------

{% set unary_funcs = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
                      'cos', 'cosh', 'exp', 'floor', 'logical_not', 'negative',
                      'sigmoid', 'sign', 'sin', 'sinh', 'sqrt', 'square', 'tan',
                      'tanh'] %}
{% for func in unary_funcs %}

@overload
def {{ func }}(x: Array0) -> Array0: ...


@overload
def {{ func }}(x: Array1[A1]) -> Array1[A1]: ...


@overload
def {{ func }}(x: Array2[A1, A2]) -> Array2[A1, A2]: ...


@overload
def {{ func }}(x: Array3[A1, A2, A3]) -> Array3[A1, A2, A3]: ...


@overload
def {{ func }}(x: Array4[A1, A2, A3, A4]) -> Array4[A1, A2, A3, A4]: ...

{% endfor %}


{% set dtype_unary_funcs = ['zeros_like', 'ones_like'] %}
{% for func in dtype_unary_funcs %}

@overload
def {{ func }}(x: Array0, dtype=...) -> Array0: ...


@overload
def {{ func }}(x: Array1[A1], dtype=...) -> Array1[A1]: ...


@overload
def {{ func }}(x: Array2[A1, A2], dtype=...) -> Array2[A1, A2]: ...


@overload
def {{ func }}(x: Array3[A1, A2, A3], dtype=...) -> Array3[A1, A2, A3]: ...


@overload
def {{ func }}(x: Array4[A1, A2, A3, A4], dtype=...) -> Array4[A1, A2, A3, A4]: ...

{% endfor %}


@overload
def round(x: Array0, decimals=...) -> Array0: ...


@overload
def round(x: Array1[A1], decimals=...) -> Array1[A1]: ...


@overload
def round(x: Array2[A1, A2], decimals=...) -> Array2[A1, A2]: ...


@overload
def round(x: Array3[A1, A2, A3], decimals=...) -> Array3[A1, A2, A3]: ...


@overload
def round(x: Array4[A1, A2, A3, A4], decimals=...) -> Array4[A1, A2, A3, A4]: ...


# I what even why would you
@overload
def sqrt(x: float) -> float: ...

# ---------- ZEROS, ONES ----------

# Can't type these properly when shape is specified as a list. :(
# But if shape is specified as an int or a tuple, we're good! :)

@overload
def zeros(shape: List, dtype=...) -> Any: ...


@overload
def zeros(shape: L0, dtype=...) -> Array0: ...


{% for i in range(1, 5) %}
{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def zeros(shape: L{{ i }}, dtype=...) -> Array{{ i }}[{{ n_any }}]: ...


@overload
def zeros(shape: Shape{{ i }}, dtype=...) -> Array{{ i }}[{{ n_any }}]: ...

{% endfor %}

# Array0 is down here because otherwise it'd match shape e.g. Tuple[Any, Any]
# https://github.com/google/pytype/issues/767
# (e.g. `dim = some_func_that_returns_any; zeros((dim, dim))` would be Array0)
@overload
def zeros(shape: Tuple[()], dtype=...) -> Array0: ...


@overload
def ones(shape: List, dtype=...) -> Any: ...


@overload
def ones(shape: L0, dtype=...) -> Array0: ...


{% for i in range(1, 5) %}
{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def ones(shape: L{{ i }}, dtype=...) -> Array{{ i }}[{{ n_any }}]: ...


@overload
def ones(shape: Shape{{ i }}, dtype=...) -> Array{{ i }}[{{ n_any }}]: ...

{% endfor %}

# See note about Array0 in `zeros`
@overload
def ones(shape: Tuple[()], dtype=...) -> Array0: ...


# ---------- REDUCTION OPERATORS ----------

{% for op in ['sum', 'mean'] %}

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
#   to anything (even `None`) produces a "The 'out' argument to jnp.{{op}} is
#   not supported" error.
@overload
def {{ op }}(
    a: Any,
    keepdims: Literal[True],
    axis=...,
    out=...,
    dtype=...
) -> Any: ...

## keepdims = False or unspecified

{% for n_axes in range(1, 5) %}

### n_axes = {{ n_axes }}

#### `axis` specified

{% for axes in reduction_axes(n_axes) %}

@overload
def {{ op }}(
    a: Array{{ axes.n_axes }}{{ axes.all_axes }},
    axis: {{ axes.reduction_axes }},
    out=..., keepdims=..., dtype=...
) -> Array{{ axes.remaining_n_axes }}{{ axes.remaining_axes }}: ...

{% endfor %}

# Fallback: `axis` not any of the above
@overload
def {{ op }}(
    a: Array{{ n_axes }}[{{ (['Any'] * n_axes)|join(', ') }}],
    axis: Any,
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def {{ op }}(
    a: Array{{ n_axes }}[{{ (['Any'] * n_axes)|join(', ') }}],
    out=..., keepdims=..., dtype=...
) -> Array0: ...

{% endfor %}

### Some weird argument like a list of arrays

@overload
def {{ op }}(
    a: Any,
    axis=...,
    out=..., keepdims=..., dtype=...
) -> Any: ...

{% endfor %}

# ---------- TRANSPOSE ----------

{% for n_axes in range(1, 5) %}

### n_axes = {{ n_axes }}

#### `axes` specified

{% for axes in transpose_axes(n_axes) %}

@overload
def transpose(
    a: Array{{ n_axes }}{{ axes.all_axes }},
    axes: {{ axes.transpose_axes }}
) -> Array{{ n_axes }}{{ axes.result_axes }}: ...

{% endfor %}

#### `axes` unspecified

{# axes = 'A1, A2, A3' #}
{% set axes = get_axis_list(n_axes) %}
{# reverse_axes = 'A3, A2, A1' #}
{% set reverse_axes = get_axis_list(n_axes, reverse=True) %}

@overload
def transpose(
    a: Array{{ n_axes }}[{{ axes }}]
) -> Array{{ n_axes }}[{{ reverse_axes }}]: ...

{% endfor %}

# ---------- EVERYTHING ELSE: UNTYPED ----------

{% for x in jnp_dir %}
{{ x }}: Any
{% endfor %}

# LINT.ThenChange(../library_stubs/third_party/py/jax/numpy/__init__.pyi)

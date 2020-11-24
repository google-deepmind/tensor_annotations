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

from tensor_annotations.jax import array0, array1, array2, array3, array4
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


# ---------- UNARY OPERATORS ----------

{% for i in range(1, 5) %}

{# type = arrayN[A1, A2, ..., An] #}
{% set type = get_jax_array_type(i) %}

@overload
def abs(x: {{ type }}) -> {{ type }}: ...


@overload
def acos(x: {{ type }}) -> {{ type }}: ...


@overload
def acosh(x: {{ type }}) -> {{ type }}: ...


@overload
def asin(x: {{ type }}) -> {{ type }}: ...


@overload
def asinh(x: {{ type }}) -> {{ type }}: ...


@overload
def atan(x: {{ type }}) -> {{ type }}: ...


@overload
def atanh(x: {{ type }}) -> {{ type }}: ...


@overload
def cos(x: {{ type }}) -> {{ type }}: ...


@overload
def cosh(x: {{ type }}) -> {{ type }}: ...


@overload
def exp(x: {{ type }}) -> {{ type }}: ...


@overload
def floor(x: {{ type }}) -> {{ type }}: ...


@overload
def logical_not(x: {{ type }}) -> {{ type }}: ...


@overload
def negative(x: {{ type }}) -> {{ type }}: ...


@overload
def ones_like(x: {{ type }}, dtype=...) -> {{ type }}: ...


@overload
def round(x: {{ type }}, decimals=...) -> {{ type }}: ...


@overload
def sigmoid(x: {{ type }}) -> {{ type }}: ...


@overload
def sign(x: {{ type }}) -> {{ type }}: ...


@overload
def sin(x: {{ type }}) -> {{ type }}: ...


@overload
def sinh(x: {{ type }}) -> {{ type }}: ...


@overload
def sqrt(x: {{ type }}) -> {{ type }}: ...


@overload
def square(x: {{ type }}) -> {{ type }}: ...


@overload
def tan(x: {{ type }}) -> {{ type }}: ...


@overload
def tanh(x: {{ type }}) -> {{ type }}: ...


@overload
def zeros_like(x: {{ type }}, dtype=...) -> {{ type }}: ...

{% endfor %}

# I what even why would you
@overload
def sqrt(x: float) -> float: ...

# ---------- ZEROS, ONES ----------

# Can't type these properly when shape is specified as a list. :(

@overload
def zeros(shape: List, dtype=...) -> Any: ...


@overload
def ones(shape: List, dtype=...) -> Any: ...

# But if shape is specified as an int or a tuple, we're good! :)

@overload
def zeros(shape: L0, dtype=...) -> array0: ...


@overload
def zeros(shape: Tuple[()], dtype=...) -> array0: ...


@overload
def ones(shape: L0, dtype=...) -> array0: ...


@overload
def ones(shape: Tuple[()], dtype=...) -> array0: ...


{% for i in range(1, 4) %}

{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def zeros(shape: L{{ i }}, dtype=...) -> array{{ i }}[{{ n_any }}]: ...


@overload
def ones(shape: L{{ i }}, dtype=...) -> array{{ i }}[{{ n_any }}]: ...


@overload
def zeros(shape: Shape{{ i }}, dtype=...) -> array{{ i }}[{{ n_any }}]: ...


@overload
def ones(shape: Shape{{ i }}, dtype=...) -> array{{ i }}[{{ n_any }}]: ...

{% endfor %}

# ---------- REDUCTION OPERATORS ----------

{% for op in ['sum', 'mean'] %}

## keepdims = True: yet be to be typed

@overload
def {{ op }}(
    a: Any,
    axis=...,
    out=...,
    keepdims: Literal[True],
    dtype=...
) -> Any: ...

## keepdims = False or unspecified

{% for n_axes in range(1, 5) %}

### n_axes = {{ n_axes }}

#### `axis` specified

{% for axes in reduction_axes(n_axes) %}

@overload
def {{ op }}(
    a: array{{ axes.n_axes }}{{ axes.all_axes }},
    axis: {{ axes.reduction_axes }},
    out=..., keepdims=..., dtype=...
) -> array{{ axes.remaining_n_axes }}{{ axes.remaining_axes }}: ...

{% endfor %}

# Fallback: `axis` not any of the above
@overload
def {{ op }}(
    a: array{{ n_axes }}[{{ (['Any'] * n_axes)|join(', ') }}],
    axis: Literal[Any],
    out=..., keepdims=..., dtype=...
) -> Any: ...

#### `axis` unspecified

@overload
def {{ op }}(
    a: array{{ n_axes }}[{{ (['Any'] * n_axes)|join(', ') }}],
    out=..., keepdims=..., dtype=...
) -> array0: ...

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
    a: array{{ n_axes }}{{ axes.all_axes }},
    axes: {{ axes.transpose_axes }}
) -> array{{ n_axes }}{{ axes.result_axes }}: ...

{% endfor %}

#### `axes` unspecified

{# axes = 'A1, A2, A3' #}
{% set axes = get_axis_list(n_axes) %}
{# reverse_axes = 'A3, A2, A1' #}
{% set reverse_axes = get_axis_list(n_axes, reverse=True) %}

@overload
def transpose(
    a: array{{ n_axes }}[{{ axes }}]
) -> array{{ n_axes }}[{{ reverse_axes }}]: ...

{% endfor %}

# ---------- EVERYTHING ELSE: UNTYPED ----------

{% for x in jnp_dir %}
{{ x }}: Any
{% endfor %}

# LINT.ThenChange(../library_stubs/third_party/py/jax/numpy/__init__.pyi)


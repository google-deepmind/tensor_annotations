# BEGIN PREAMBLE
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
"""TensorFlow stubs.

NOTE: This file is generated from templates/tensorflow.pyi
      using a Google-internal tool.
"""

# BEGIN: tensor_annotations annotations
from typing import Any, TypeVar, Tuple, overload
from typing_extensions import Literal

from tensor_annotations.axes import Axis
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)
A5 = TypeVar('A5', bound=Axis)
# This alias makes the meaning clearer in code.
# Unfortunately, it still shows up as 'Any' in pytype output.
AnyDType = Any
DT = TypeVar('DT')

TRUE = Literal[True]
FALSE = Literal[False]

LN1 = Literal[-1]
L0 = Literal[0]
L1 = Literal[1]
L2 = Literal[2]
L3 = Literal[3]
L4 = Literal[4]

Shape1 = Tuple[int]
Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Shape4 = Tuple[int, int, int, int]
Shape5 = Tuple[int, int, int, int, int]
# END: tensor_annotations annotations

# END PREAMBLE


# ---------- UNARY OPERATORS ----------

{% set unary_funcs = ['abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
                      'cos', 'cosh', 'erf', 'erfc', 'erfinv', 'exp', 'expm1',
                      'floor', 'is_finite', 'is_inf', 'is_nan', 'lbeta',
                      'lgamma', 'log', 'log1p', 'logical_not', 'log_sigmoid',
                      'ndtri', 'negative', 'reciprocal', 'reciprocal_no_nan',
                      'rint', 'round', 'rsqrt', 'sigmoid', 'sign', 'sin', 'sinh',
                      'softplus', 'softsign', 'sqrt', 'square', 'tan', 'tanh'] %}
{% for func in unary_funcs %}

@overload
def {{ func }}(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...


@overload
def {{ func }}(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...


@overload
def {{ func }}(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...


@overload
def {{ func }}(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...


@overload
def {{ func }}(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...


@overload
def {{ func }}(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...


@overload
def {{ func }}(x, name=...) -> Any: ...

{% endfor %}


{% set dtype_unary_funcs = ['zeros_like', 'ones_like'] %}
{% for func in dtype_unary_funcs %}

@overload
def {{ func }}(input: Tensor1[AnyDType, A1], dtype=..., name=...) -> Tensor1[AnyDType, A1]: ...


@overload
def {{ func }}(input: Tensor2[AnyDType, A1, A2], dtype=..., name=...) -> Tensor2[AnyDType, A1, A2]: ...


@overload
def {{ func }}(input: Tensor3[AnyDType, A1, A2, A3], dtype=..., name=...) -> Tensor3[AnyDType, A1, A2, A3]: ...


@overload
def {{ func }}(input: Tensor4[AnyDType, A1, A2, A3, A4], dtype=..., name=...) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...


@overload
def {{ func }}(input: Tensor5[AnyDType, A1, A2, A3, A4, A5], dtype=..., name=...) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...


@overload
def {{ func }}(input, dtype=..., name=...) -> Any: ...
{% endfor %}


# ---------- ZEROS, ONES ----------

# Can't type these properly when shape is specified as a list. :(
# But if shape is specified as an int or a tuple, we're good! :)

{% for i in range(1, 6) %}
{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def zeros(shape: Shape{{ i }}, dtype=..., name=...) -> Tensor{{ i }}[AnyDType, {{ n_any }}]: ...

{% endfor %}

# Tensor0 is down here because otherwise it'd match shape e.g. Tuple[Any, Any]
# https://github.com/google/pytype/issues/767
# (e.g. `dim = tf.shape_as_list(x); tf.zeros((dim, dim))` would be Tensor0)
@overload
def zeros(shape: Tuple[()], dtype=..., name=...) -> Tensor0[AnyDType]: ...

@overload
def zeros(shape, dtype=..., name=...) -> AnyDType: ...


{% for i in range(1, 6) %}
{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def ones(shape: Shape{{ i }}, dtype=..., name=...) -> Tensor{{ i }}[AnyDType, {{ n_any }}]: ...

{% endfor %}

# See note about Tensor0 in `zeros`
@overload
def ones(shape: Tuple[()], dtype=..., name=...) -> Tensor0[AnyDType]: ...

@overload
def ones(shape, dtype=..., name=...) -> AnyDType: ...


# ---------- REDUCTION OPERATORS ----------

{% for op in ['reduce_all', 'reduce_any', 'reduce_logsumexp', 'reduce_max',
              'reduce_mean', 'reduce_min', 'reduce_prod', 'reduce_sum'] %}

# axis specified

{% for n_axes in range(1, 6) %}
{% for axes in reduction_axes(n_axes) %}

@overload
def {{ op }}(input_tensor: Tensor{{ axes.n_axes }}[DT, {{ axes.all_axes }}],
               axis: {{ axes.reduction_axes }}, name=...) -> Tensor{{ axes.remaining_n_axes }}[DT{{ axes.remaining_axes }}]: ...

{% endfor %}
{% endfor %}

# axis unspecified, keepdims=True

{% for n_axes in range(1, 6) %}
{% set all_axes = get_axis_list(n_axes, reverse=True) %}

@overload
def {{ op }}(input_tensor: Tensor{{ n_axes }}[DT, {{ all_axes }}],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor{{ n_axes }}[DT, {{ all_axes }}]: ...

{% endfor %}

@overload
def {{ op }}(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...

{% endfor %}

# ---------- TRANSPOSE ----------

#### `perm` unspecified

{% for n_axes in range(2, 6) %}
{# axes = 'A1, A2, A3' #}
{% set axes = get_axis_list(n_axes) %}
{# reverse_axes = 'A3, A2, A1' #}
{% set reverse_axes = get_axis_list(n_axes, reverse=True) %}

@overload
def transpose(a: Tensor{{ n_axes }}[DT, {{ axes }}], name=...) -> Tensor{{ n_axes }}[DT, {{ reverse_axes }}]: ...

{% endfor %}

#### `perm` specified

{% for n_axes in range(2, 6) %}
{% for axes in transpose_axes(n_axes) %}

@overload
def transpose(a: Tensor{{ n_axes }}[DT, {{ axes.all_axes }}], perm: {{ axes.transpose_axes }},
              name=...) -> Tensor{{ n_axes }}[DT, {{ axes.result_axes }}]: ...

{% endfor %}
{% endfor %}

@overload
def transpose(a, perm=..., conjugate=..., name=...) -> Any: ...

# ---------- MATMUL ----------

@overload
def matmul(
    a: Tensor2[AnyDType, A1, A2],
    b: Tensor2[AnyDType, A2, A3],
    name=...,
) -> Tensor2[AnyDType, A1, A3]: ...

@overload
def matmul(
    a: Tensor2[AnyDType, A1, A2],
    b: Tensor2[AnyDType, A1, A3],
    transpose_a: TRUE,
    name=...
) -> Tensor2[AnyDType, A2, A3]: ...

@overload
def matmul(
    a: Tensor2[AnyDType, A1, A2],
    b: Tensor2[AnyDType, A3, A2],
    transpose_b: TRUE,
    name=...
) -> Tensor2[AnyDType, A1, A3]: ...

@overload
def matmul(
    a: Tensor3[AnyDType, A1, A2, A3],
    b: Tensor2[AnyDType, A3, A4],
    name=...
) -> Tensor3[AnyDType, A1, A2, A4]: ...

@overload
def matmul(
    a: Tensor3[AnyDType, A1, A2, A3],
    b: Tensor2[AnyDType, A4, A3],
    transpose_b: TRUE,
    name=...
) -> Tensor3[AnyDType, A4, A2, A3]: ...

@overload
def matmul(
    a, b,
    transpose_a=..., transpose_b=...,
    adjoint_a=..., adjoint_b=...,
    a_is_sparse=..., b_is_sparse=...,
    name=...
) -> Any: ...

# LINT.ThenChange(
#     ../library_stubs/third_party/py/tensorflow/__init__.pyi,
#     ../library_stubs/third_party/py/tensorflow/math/__init__.pyi
# )

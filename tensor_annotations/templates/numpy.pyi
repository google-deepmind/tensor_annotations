# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Stubs for np.*

NOTE: This file is generated from templates/numpy.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_numpy_library_template.py
"""

# We use old-style annotations - `List` and `Tuple` rather than `list` and
# `tuple` - because we want to be compatible with older versions of Python.
from typing import Any, List, overload, Tuple

import tensor_annotations.numpy as tnp
from tensor_annotations.numpy import Array1, Array2, Array3, Array4
from tensor_annotations.axes import Axis

AnyDType = Any
DT = TypeVar('DT', bound=tnp.DType)

A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)

Shape0 = Tuple[()]
Shape1 = Tuple[int]
Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Shape4 = Tuple[int, int, int, int]

# ---------- ZEROS_LIKE, ONES_LIKE ----------

{% set dtype_unary_funcs = ['zeros_like', 'ones_like'] %}
{% for func in dtype_unary_funcs %}

@overload
def {{ func }}(x: Array1[DT, A1], dtype=...) -> Array1[DT, A1]: ...


@overload
def {{ func }}(x: Array2[DT, A1, A2], dtype=...) -> Array2[DT, A1, A2]: ...


@overload
def {{ func }}(x: Array3[DT, A1, A2, A3], dtype=...) -> Array3[DT, A1, A2, A3]: ...


@overload
def {{ func }}(x: Array4[DT, A1, A2, A3, A4], dtype=...) -> Array4[DT, A1, A2, A3, A4]: ...

{% endfor %}

# ---------- ZEROS, ONES ----------

# Can't type these properly when shape is specified as a list. :(
# But if shape is specified as an int or a tuple, we're good! :)

@overload
def zeros(shape: List, dtype=...) -> Any: ...


@overload
def zeros(shape: int, dtype=...) -> Array1[AnyDType, Any]: ...


{% for i in range(1, 5) %}
{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def zeros(shape: Shape{{ i }}, dtype=...) -> Array{{ i }}[AnyDType, {{ n_any }}]: ...

{% endfor %}


@overload
def ones(shape: List, dtype=...) -> Any: ...


@overload
def ones(shape: int, dtype=...) -> Array1[AnyDType, Any]: ...


{% for i in range(1, 5) %}
{% set n_any = (['Any'] * i)|join(', ') %}

@overload
def ones(shape: Shape{{ i }}, dtype=...) -> Array{{ i }}[AnyDType, {{ n_any }}]: ...

{% endfor %}

# ---------- EVERYTHING ELSE: UNTYPED ----------

# We need to special-case this because the type of the `dtype` attribute of a
# `jax.Array` (that is, JAX's built-in array type - _not_ a Tensor annotations
# array class) is actually `np.dtype` - so ideally it should be something more
# than just `Any`.
class dtype: pass

{% for x in np_dir %}
{{ x }}: Any
{% endfor %}

# LINT.ThenChange(../library_stubs/third_party/py/numpy/__init__.pyi)

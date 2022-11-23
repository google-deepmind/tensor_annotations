# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Stubs for jax.*

We also need to provide stubs for jax.Array in here to avoid breaking
code which doesn't use tensor_annotations annotations. Le sigh.

NOTE: This file is generated from templates/jax.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_jax_library_template.py
"""

from typing import Any

_sentinel: int


class Array:

  shape: tuple[int, ...]

  item: Any

  def astype(self, dtype) -> 'Array':
    ...

  def __add__(self, other) -> 'Array':
    ...

  def __radd__(self, other) -> 'Array':
    ...

  def __sub__(self, other) -> 'Array':
    ...

  def __rsub__(self, other) -> 'Array':
    ...

  def __mul__(self, other) -> 'Array':
    ...

  def __rmul__(self, other) -> 'Array':
    ...

  def __floordiv__(self, other) -> 'Array':
    ...

  def __truediv__(self, other) -> 'Array':
    ...

  def __pow__(self, other) -> 'Array':
    ...

  def __matmul__(self, other) -> 'Array':
    ...

  @property
  def T(self) -> 'Array':
    ...

  def __getitem__(self, key) -> 'Array':
    ...

  @property
  def at(self) -> '_IndexUpdateHelper':
    ...


class _IndexUpdateHelper:

  def __getitem__(self, key) -> '_IndexUpdateRef':
    ...


class _IndexUpdateRef:

  def set(self, value) -> Array:
    ...


{% for x in jax_dir %}
{{ x }}: Any
{% endfor %}

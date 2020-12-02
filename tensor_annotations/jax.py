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
"""Custom tensor classes for JAX supporting shape parameterisation.

Note that these should only be used for the purposes of type annotation and
should never be instantiated. (Certain IDEs may also use these for
autocompletion, too.)

Type annotations for these classes are maintained in a separate stubs file,
`jax.pyi`.
"""

# LINT.IfChange

from typing import Any, Generic, TypeVar

from tensor_annotations import axes

A1 = TypeVar('A1', bound=axes.Axis)
A2 = TypeVar('A2', bound=axes.Axis)
A3 = TypeVar('A3', bound=axes.Axis)
A4 = TypeVar('A4', bound=axes.Axis)

# Necessary to silence warnings about arrayN not being a valid name.
# pylint: disable=invalid-name


class Array0:
  """A scalar - produced by e.g. jnp.sum(jnp.zeros((2, 3)))."""

  def __new__(cls, *args, **kwargs):
    raise TypeError('tensor_annotations tensors should not be instantiated')

  # These are necessary so that type checkers know we have these methods.
  __abs__: Any
  __add__: Any
  __add__: Any
  __floordiv__: Any
  __ge__: Any
  __gt__: Any
  __le__: Any
  __lt__: Any
  __mul__: Any
  __neg__: Any
  __neg__: Any
  __pos__: Any
  __pow__: Any
  __rmul__: Any
  __sub__: Any
  __truediv__: Any
  shape: Any
  type: Any
  reshape: Any


class Array1(Generic[A1]):
  """A tensor of rank 1."""

  def __new__(cls, *args, **kwargs):
    raise TypeError('tensor_annotations tensors should not be instantiated')

  # These are necessary so that type checkers know we have these methods.
  __abs__: Any
  __add__: Any
  __add__: Any
  __floordiv__: Any
  __getitem__: Any
  __setitem__: Any
  __ge__: Any
  __gt__: Any
  __le__: Any
  __lt__: Any
  __mul__: Any
  __neg__: Any
  __neg__: Any
  __pos__: Any
  __pow__: Any
  __rmul__: Any
  __sub__: Any
  __truediv__: Any
  shape: Any
  type: Any
  reshape: Any


class Array2(Generic[A1, A2]):
  """A tensor of rank 2."""

  def __new__(cls, *args, **kwargs):
    raise TypeError('tensor_annotations tensors should not be instantiated')

  # These are necessary so that type checkers know we have these methods.
  __abs__: Any
  __add__: Any
  __add__: Any
  __floordiv__: Any
  __getitem__: Any
  __setitem__: Any
  __ge__: Any
  __gt__: Any
  __le__: Any
  __lt__: Any
  __mul__: Any
  __neg__: Any
  __neg__: Any
  __pos__: Any
  __pow__: Any
  __rmul__: Any
  __sub__: Any
  __truediv__: Any
  shape: Any
  type: Any
  reshape: Any


class Array3(Generic[A1, A2, A3]):
  """A tensor of rank 3."""

  def __new__(cls, *args, **kwargs):
    raise TypeError('tensor_annotations tensors should not be instantiated')

  # These are necessary so that type checkers know we have these methods.
  __abs__: Any
  __add__: Any
  __add__: Any
  __floordiv__: Any
  __getitem__: Any
  __setitem__: Any
  __ge__: Any
  __gt__: Any
  __le__: Any
  __lt__: Any
  __mul__: Any
  __neg__: Any
  __neg__: Any
  __pos__: Any
  __pow__: Any
  __rmul__: Any
  __sub__: Any
  __truediv__: Any
  shape: Any
  type: Any
  reshape: Any


class Array4(Generic[A1, A2, A3, A4]):
  """A tensor of rank 4."""

  def __new__(cls, *args, **kwargs):
    raise TypeError('tensor_annotations tensors should not be instantiated')

  # These are necessary so that type checkers know we have these methods.
  __abs__: Any
  __add__: Any
  __add__: Any
  __floordiv__: Any
  __getitem__: Any
  __setitem__: Any
  __ge__: Any
  __gt__: Any
  __le__: Any
  __lt__: Any
  __mul__: Any
  __neg__: Any
  __neg__: Any
  __pos__: Any
  __pow__: Any
  __rmul__: Any
  __sub__: Any
  __truediv__: Any
  shape: Any
  type: Any
  reshape: Any

# LINT.ThenChange(jax.pyi)

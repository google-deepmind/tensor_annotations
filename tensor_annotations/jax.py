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
A5 = TypeVar('A5', bound=axes.Axis)
A6 = TypeVar('A6', bound=axes.Axis)
A7 = TypeVar('A7', bound=axes.Axis)
A8 = TypeVar('A8', bound=axes.Axis)


class _ArrayBase:
  """Base class for ArrayN classes containing common methods and attributes."""

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


class Array0(_ArrayBase):
  """A scalar - produced by e.g. jnp.sum(jnp.zeros((2, 3)))."""
  pass


class Array1(Generic[A1], _ArrayBase):
  """A tensor of rank 1."""
  pass


class Array2(Generic[A1, A2], _ArrayBase):
  """A tensor of rank 2."""
  pass


class Array3(Generic[A1, A2, A3], _ArrayBase):
  """A tensor of rank 3."""
  pass


class Array4(Generic[A1, A2, A3, A4], _ArrayBase):
  """A tensor of rank 4."""
  pass


class Array5(Generic[A1, A2, A3, A4, A5], _ArrayBase):
  """A tensor of rank 4."""
  pass


class Array6(Generic[A1, A2, A3, A4, A5, A6], _ArrayBase):
  """A tensor of rank 4."""
  pass


class Array7(Generic[A1, A2, A3, A4, A5, A6, A7], _ArrayBase):
  """A tensor of rank 4."""
  pass


class Array8(Generic[A1, A2, A3, A4, A5, A6, A7, A8], _ArrayBase):
  """A tensor of rank 4."""
  pass
# LINT.ThenChange(jax.pyi)

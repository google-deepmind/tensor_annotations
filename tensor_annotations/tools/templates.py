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
"""Helper functions for template rendering."""

import collections
import itertools


ReductionAxes = collections.namedtuple('ReductionAxes',
                                       ['n_axes', 'all_axes', 'reduction_axes',
                                        'remaining_n_axes', 'remaining_axes'])

TransposeAxes = collections.namedtuple('TransposeAxes',
                                       ['n_axes', 'all_axes', 'transpose_axes',
                                        'result_axes'])


def axis_list(n_axes: int, reverse=False) -> str:
  """Returns a comma-separated list of axis TypeVar short names.

  Args:
    n_axes: Maximum number of axes to include in the list.
            n_axes=1 -> 'A1', n_axes=2 -> 'A1, A2', etc.
    reverse: If False, the returned list starts from A1 and counts up to An.
             If True, starts at An and counts down to A1.
  Returns:
    A string containing the list of axes.

  For example, get_axis_list(2) -> 'A1, A2'.
  """
  axes = range(1, n_axes + 1)
  if reverse:
    axes = reversed(axes)
  return ', '.join(f'A{i}' for i in axes)


def jax_array_type(n_axes: int) -> str:
  """Returns the generic JAX array type, parameterised by a number of axes.

  For example, get_jax_array_type(2) -> 'Array2[A1, A2]'.

  Args:
    n_axes: Rank of array type to return.
  Returns:
    A string containing array type.
  """
  return f'Array{n_axes}[{axis_list(n_axes)}]'


# TODO: remove `reverse` argument
def transpose_axes(n_axes: int, reverse: bool = False):
  """A generator that yields input and output axes of transpose.

  Args:
    n_axes: Rank of array whose possible transposes to consider.
    reverse: TODO

  Yields:
    A `TransposeAxes` object for each possible transpose.

  For example, calculate_transpose_axes(2) would yield `TransposeAxes` objects
  encoding:
    Transpose shape [A1, A2] with axes=[0, 1] -> Shape[A1, A2]
                                       [1, 0] -> Shape[A2, A1]
  """
  assert n_axes >= 1

  # [A1, A2, ..., An]
  all_axes = list(range(1, n_axes + 1))
  all_axes_str = [f'A{i}' for i in all_axes]
  if reverse:
    all_axes_str = reversed(all_axes_str)
  all_axes_str = ', '.join(all_axes_str)
  all_axes_str = '[' + all_axes_str + ']'

  for transpose_axes in itertools.permutations(range(n_axes)):
    transpose_axes_str = (f'L{i}' for i in transpose_axes)
    transpose_axes_str = ', '.join(transpose_axes_str)
    transpose_axes_str = f'Tuple[{transpose_axes_str}]'

    if reverse:
      result_axes = (all_axes[n_axes-1-i] for i in transpose_axes)
    else:
      result_axes = (all_axes[i] for i in transpose_axes)
    if result_axes:
      result_axes_str = (f'A{i}' for i in result_axes)
      result_axes_str = ', '.join(result_axes_str)
      result_axes_str = '[' + result_axes_str + ']'
    else:
      result_axes_str = ''

    yield TransposeAxes(n_axes=n_axes,
                        all_axes=all_axes_str,
                        transpose_axes=transpose_axes_str,
                        result_axes=result_axes_str)


# TODO: Remove `reverse` and `single_reduction_axis_only`
def reduction_axes(n_axes: int, reverse: bool = False,
                   single_reduction_axis_only: bool = False):
  """A generator that yields input and output axes of reduction operations.

  Args:
    n_axes: Rank of array whose possible reductions to consider.
    reverse: TODO
    single_reduction_axis_only: TODO

  Yields:
    A `ReductionAxes` object for each possible reduction.

  For example, calculate_reduction_axes(2) would yield `ReductionAxes` objects
  encoding:
    Reduce shape [A1, A2] over axes    0 -> shape[A2]
                                       1 -> shape[A1]
                                    0, 1 -> shape[]
                                    1, 0 -> shape[]
  """
  assert n_axes >= 1

  # [A1, A2, ..., An]
  all_axes_str = [f'A{i}' for i in range(1, n_axes + 1)]
  if reverse: all_axes_str = reversed(all_axes_str)
  all_axes_str = ', '.join(all_axes_str)
  all_axes_str = '[' + all_axes_str + ']'

  if single_reduction_axis_only:
    n_reduction_axes_iter = [1]
  else:
    n_reduction_axes_iter = range(1, n_axes + 1)
  for n_reduction_axes in n_reduction_axes_iter:
    for reduction_axes in itertools.permutations(range(n_axes),
                                                 n_reduction_axes):
      if len(reduction_axes) == 1:
        reduction_axes_str = f'L{reduction_axes[0]}'
      else:
        reduction_axes_str = (f'L{i}' for i in reduction_axes)
        reduction_axes_str = ', '.join(reduction_axes_str)
        reduction_axes_str = f'Tuple[{reduction_axes_str}]'

      remaining_axes = set(range(n_axes)) - set(reduction_axes)
      remaining_axes = sorted(tuple(remaining_axes))
      remaining_n_axes = len(remaining_axes)
      if remaining_axes:
        if reverse:
          remaining_axes_str = (f'A{n_axes - i}' for i in remaining_axes)
        else:
          remaining_axes_str = (f'A{i + 1}' for i in remaining_axes)
        remaining_axes_str = ', '.join(remaining_axes_str)
        remaining_axes_str = '[' + remaining_axes_str + ']'
      else:
        remaining_axes_str = ''

      yield ReductionAxes(n_axes,
                          all_axes_str,
                          reduction_axes_str,
                          remaining_n_axes,
                          remaining_axes_str)

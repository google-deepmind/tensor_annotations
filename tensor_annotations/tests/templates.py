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
"""Tests for tools/templates.py."""

import unittest

from tensor_annotations.tools import templates


class TemplateTests(unittest.TestCase):

  def test_axis_list(self):
    self.assertEqual(
        templates.axis_list(n_axes=1),
        'A1'
    )
    self.assertEqual(
        templates.axis_list(n_axes=2),
        'A1, A2'
    )
    self.assertEqual(
        templates.axis_list(n_axes=1, reverse=True),
        'A1'
    )
    self.assertEqual(
        templates.axis_list(n_axes=2, reverse=True),
        'A2, A1'
    )

  def test_jax_array_type(self):
    self.assertEqual(
        'Array1[A1]',
        templates.jax_array_type(n_axes=1)
    )
    self.assertEqual(
        'Array2[A1, A2]',
        templates.jax_array_type(n_axes=2)
    )

  def test_transpose_axes(self):
    self.assertEqual(
        list(templates.transpose_axes(1)),
        [
            templates.TransposeAxes(
                n_axes=1,
                all_axes='[A1]',
                transpose_axes='Tuple[L0]',
                result_axes='[A1]'
            )
        ]
    )

    self.assertEqual(
        list(templates.transpose_axes(2)),
        [
            templates.TransposeAxes(
                n_axes=2,
                all_axes='[A1, A2]',
                transpose_axes='Tuple[L0, L1]',
                result_axes='[A1, A2]'
            ),
            templates.TransposeAxes(
                n_axes=2,
                all_axes='[A1, A2]',
                transpose_axes='Tuple[L1, L0]',
                result_axes='[A2, A1]'
            ),
        ]
    )

  def test_reduction_axes(self):
    self.assertEqual(
        list(templates.reduction_axes(1)),
        [
            templates.ReductionAxes(
                n_axes=1,
                all_axes='[A1]',
                reduction_axes='L0',
                remaining_n_axes=0,
                remaining_axes=''
            )
        ]
    )

    self.assertEqual(
        list(templates.reduction_axes(2)),
        [
            templates.ReductionAxes(
                n_axes=2,
                all_axes='[A1, A2]',
                reduction_axes='L0',
                remaining_n_axes=1,
                remaining_axes='[A2]'
            ),
            templates.ReductionAxes(
                n_axes=2,
                all_axes='[A1, A2]',
                reduction_axes='L1',
                remaining_n_axes=1,
                remaining_axes='[A1]'
            ),
            templates.ReductionAxes(
                n_axes=2,
                all_axes='[A1, A2]',
                reduction_axes='Tuple[L0, L1]',
                remaining_n_axes=0,
                remaining_axes=''
            ),
        ]
    )


if __name__ == '__main__':
  unittest.main()

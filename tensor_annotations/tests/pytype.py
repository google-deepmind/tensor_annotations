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
"""Tests to confirm that Pytype helpers function correctly."""

import textwrap
import unittest

from tensor_annotations.tests import utils


_PREAMBLE = """
from typing import NewType

from tensor_annotations.jax import Array2
from tensor_annotations import axes

A1 = NewType('A1', axes.Axis)
A2 = NewType('A2', axes.Axis)
"""


class PytypeTests(unittest.TestCase):

  def testSimpleCorrectExample_PassesPytype(self):
    code = """
      def foo(x: Array2[A1, A2]):
        pass
      x: Array2[A1, A2] = Array2()
      foo(x)
    """
    code = _PREAMBLE + textwrap.dedent(code)
    utils.assert_pytype_succeeds(code)

  def testSimpleIncorrectExample_FailsPytype(self):
    code = """
      def foo(x: Array2[A1, A2]):
        pass
      x: Array2[A2, A1] = Array2()
      foo(x)
    """
    code = _PREAMBLE + textwrap.dedent(code)
    utils.assert_pytype_fails(code)


if __name__ == '__main__':
  unittest.main()

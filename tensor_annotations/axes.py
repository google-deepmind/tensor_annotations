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
"""Canonical source for common axis types.

We need to make sure that `Batch` in one module means the same as `Batch` in
another module. Since we verify equality of axis types based on the identity of
the type, that means that both modules need to import and use the same `Batch`
type. We therefore provide this file as the canonical reference for axis types
that are likely to be used widely.
"""


class Axis:
  """Base type for axis annotations.

  User-defined axis types should subclass this.
  """
  pass


# These could be more compactly specified with typing.NewType,
# but pytype doesn't currently support NewType in stubs:
# https://github.com/google/pytype/issues/597


class Batch(Axis):
  pass


class Channels(Axis):
  pass


class Features(Axis):
  pass


class Time(Axis):
  pass


class Height(Axis):
  pass


class Width(Axis):
  pass


class Depth(Axis):
  pass

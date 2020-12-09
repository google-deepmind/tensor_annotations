#!/usr/bin/env python
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
"""Install script for JAX stubs."""

import os
import pathlib
import shutil
import tempfile

import setuptools


# Note: Copybara takes care of moving files to 'jax-stubs/'.

# We want to install stubs under a package `jax-stubs`, but I haven't
# figured out how to do that unless the stubs are in a subfolder `jax-stubs`.
# So we rearrange things a bit in a temporary directory.
setup_dir = pathlib.Path(__file__).absolute().parent
with tempfile.TemporaryDirectory() as tmp_dir:
  tmp_dir = pathlib.Path(tmp_dir)
  stubs_dir = tmp_dir / 'jax-stubs'
  (stubs_dir / 'numpy').mkdir(parents=True)
  shutil.copy(setup_dir / '__init__.pyi', stubs_dir)
  shutil.copy(setup_dir / 'numpy' / '__init__.pyi', stubs_dir / 'numpy')

  os.chdir(tmp_dir)

  setuptools.setup(
      name='jax-stubs',
      version='1.0',
      description=('Type stubs for JAX.'),
      packages=['jax-stubs'],
      package_data={'jax-stubs': ['__init__.pyi', 'numpy/__init__.pyi']},
      include_package_data=True,
  )

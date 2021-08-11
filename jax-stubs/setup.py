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

setuptools.setup(
  name='tensor-annotations-jax-stubs',
  version='1.0.0',
  description='Shape-aware type stubs for JAX.',
  long_description='Shape-aware types stubs for JAX. See the `tensor-annotations` package.',
  long_description_content_type='text/markdown',
  url='https://github.com/deepmind/tensor_annotations',
  packages=['jax-stubs'],
  package_data={'jax-stubs': ['*.pyi', '*/*.pyi']},
  install_requires=['tensor-annotations'],
)

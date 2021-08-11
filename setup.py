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
"""Install script."""

import glob
import os
import pathlib
import setuptools

stub_files = (glob.glob('tensor_annotations/*.pyi') +
              glob.glob('tensor_annotations/library_stubs/**/*.pyi',
                        recursive=True))
# package_data expects paths to be relative to the package directory, so strip
# 'tensor_annotations/' from start of paths
stub_files = [
    os.path.join(*pathlib.Path(path).parts[1:])
    for path in stub_files
]

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='tensor_annotations',
    version='1.0.0',
    description=('Enables annotations of tensor shapes in numerical computing '
                 'libraries. Includes type stubs for TensorFlow and JAX '
                 'describing how library functions change shapes.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/deepmind/tensor_annotations',
    # Copybara takes care of moving files to 'tensor_annotations/'
    packages=[
        'tensor_annotations',
        'tensor_annotations/experimental',
        'tensor_annotations/tests',
    ],
    package_data={'tensor_annotations': stub_files + ['py.typed']},
    extras_require={'dev': [
        'absl-py',
        'pytype',
    ]})

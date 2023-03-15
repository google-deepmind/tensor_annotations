# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Render Jinja template for NumPy library type stubs."""

from absl import app
import jinja2
import numpy as np
from tensor_annotations.tools import templates


_NUMPY_TEMPLATE_PATH = 'templates/numpy.pyi'
_NUMPY_STUBS_PATH = 'library_stubs/third_party/py/numpy/__init__.pyi'


def main(argv):
  del argv

  with open(_NUMPY_TEMPLATE_PATH, 'r') as f:
    lines = f.readlines()
  # Strip IfChange/ThenChange lines.
  lines = [l for l in lines if not l.startswith('# LINT')]

  numpy_template = jinja2.Template(
      ''.join(lines),
      extensions=['jinja2.ext.do'],
  )
  numpy_template.globals['reduction_axes'] = templates.reduction_axes
  numpy_template.globals['transpose_axes'] = templates.transpose_axes
  numpy_template.globals['get_axis_list'] = templates.axis_list

  # We need to make sure that the library functions we _haven't_ annotated
  # are still present in the type stubs or the type checker will think they
  # don't exist at all. We do this in a bit of a hacky way: enumerating through
  # `dir(np)` and adding an `Any` annotation for everything we find that's
  # not currently annotated.
  current_stubs = open(_NUMPY_STUBS_PATH).read()
  np_dir = []
  for x in dir(np):
    if (x.startswith('_')
        or f'def {x}(' in current_stubs
        or f'class {x}:' in current_stubs):
      continue
    np_dir.append(x)

  with open(_NUMPY_STUBS_PATH, 'w') as f:
    f.write(numpy_template.render(np_dir=np_dir))


if __name__ == '__main__':
  app.run(main)

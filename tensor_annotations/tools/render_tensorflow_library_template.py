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
"""Render Jinja template for TensorFlow library type stubs."""

from absl import app
import jinja2
from tensor_annotations.tools import templates


_TEMPLATE_PATH = 'templates/tensorflow.pyi'
_STUBS_PATH = 'tensorflow_stubs.pyi'


def main(argv):
  del argv

  with open(_TEMPLATE_PATH, 'r') as f:
    lines = f.readlines()
  # Strip IfChange/ThenChange lines.
  lines = [l for l in lines if not l.startswith('# LINT')]

  template = jinja2.Template(''.join(lines), extensions=['jinja2.ext.do'])
  template.globals['reduction_axes'] = templates.reduction_axes
  template.globals['transpose_axes'] = templates.transpose_axes
  template.globals['get_axis_list'] = templates.axis_list

  with open(_STUBS_PATH, 'w') as f:
    f.write(template.render())


if __name__ == '__main__':
  app.run(main)

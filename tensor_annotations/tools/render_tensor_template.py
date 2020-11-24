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
"""Render Jinja template for custom tensor class type stubs."""

from absl import app
from absl import flags
import jinja2
from tensor_annotations.tools import templates


flags.DEFINE_string('template', default=None, help='Template file')
flags.DEFINE_string('out', default=None, help='Output file')
flags.DEFINE_string('vars', default=None, help='A comma-separated list of '
                                               'template substitutions, '
                                               'e.g. foo=1,bar=2')
FLAGS = flags.FLAGS


def main(argv):
  del argv

  with open(FLAGS.template, 'r') as f:
    lines = f.readlines()
  # Strip IfChange/ThenChange lines.
  lines = [l for l in lines if not l.startswith('# LINT')]

  template = jinja2.Template(''.join(lines), extensions=['jinja2.ext.do'])
  template.globals['reduction_axes'] = templates.reduction_axes

  substitutions = {}
  if FLAGS.vars:
    for kv in FLAGS.vars.split(','):
      k, v = kv.split('=')
      substitutions[k] = v

  with open(FLAGS.out, 'w') as f:
    f.write(template.render(**substitutions))


if __name__ == '__main__':
  app.run(main)

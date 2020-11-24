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
"""Render Jinja template for JAX library type stubs."""

from absl import app
import jax.numpy as jnp
import jinja2
from tensor_annotations.tools import templates


_TEMPLATE_PATH = 'templates/jax.pyi'
_STUBS_PATH = 'library_stubs/third_party/py/jax/numpy/__init__.pyi'


def main(argv):
  del argv

  with open(_TEMPLATE_PATH, 'r') as f:
    lines = f.readlines()
  # Strip IfChange/ThenChange lines.
  lines = [l for l in lines if not l.startswith('# LINT')]

  template = jinja2.Template(''.join(lines), extensions=['jinja2.ext.do'])
  template.globals['reduction_axes'] = templates.reduction_axes
  template.globals['transpose_axes'] = templates.transpose_axes
  template.globals['get_jax_array_type'] = templates.jax_array_type
  template.globals['get_axis_list'] = templates.axis_list

  # We need to make sure that the library functions we _haven't_ annotated
  # are still present in the type stubs or the type checker will think they
  # don't exist at all. We do this in a bit of a hacky way: enumerating through
  # `dir(jnp)` and adding an `Any` annotation for everything we find that's
  # not currently annotated.
  current_stubs = open(_STUBS_PATH).read()
  jnp_dir = []
  for x in dir(jnp):
    if (x.startswith('_')
        or f'def {x}(' in current_stubs
        or f'class {x}:' in current_stubs):
      continue
    jnp_dir.append(x)

  with open(_STUBS_PATH, 'w') as f:
    f.write(template.render(jnp_dir=jnp_dir))


if __name__ == '__main__':
  app.run(main)

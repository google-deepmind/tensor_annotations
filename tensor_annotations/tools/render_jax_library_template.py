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
import jax
import jax.numpy as jnp
import jinja2
from tensor_annotations.tools import templates


_JAX_TEMPLATE_PATH = 'templates/jax.pyi'
_JAX_NUMPY_TEMPLATE_PATH = 'templates/jax_numpy.pyi'
_JAX_STUBS_PATH = 'library_stubs/third_party/py/jax/__init__.pyi'
_JAX_NUMPY_STUBS_PATH = 'library_stubs/third_party/py/jax/numpy/__init__.pyi'


def main(argv):
  del argv

  # ===== Render stubs for jax.* =====

  # Currently we just use `Any`` for everything in jax.*

  with open(_JAX_TEMPLATE_PATH, 'r') as f:
    lines = f.readlines()
  jax_template = jinja2.Template(
      ''.join(lines),
      extensions=['jinja2.ext.do'],
  )
  jax_dir = dir(jax)
  # We _don't_ want to stub `jax.numpy` as `Any`, because it would prevent
  # our stubs for jax.numpy.* being used.
  jax_dir.remove('numpy')
  # `jax.Array` is actually an important type, so we've added it as a class
  # manually in the template, and don't need to stub it as `Any`.
  jax_dir.remove('Array')
  with open(_JAX_STUBS_PATH, 'w') as f:
    f.write(jax_template.render(jax_dir=jax_dir))

  # ===== Render stubs for jax.numpy.* =====

  with open(_JAX_NUMPY_TEMPLATE_PATH, 'r') as f:
    lines = f.readlines()
  # Strip IfChange/ThenChange lines.
  lines = [l for l in lines if not l.startswith('# LINT')]

  jax_numpy_template = jinja2.Template(
      ''.join(lines),
      extensions=['jinja2.ext.do'],
  )
  jax_numpy_template.globals['reduction_axes'] = templates.reduction_axes
  jax_numpy_template.globals['transpose_axes'] = templates.transpose_axes
  jax_numpy_template.globals['get_jax_array_type'] = templates.jax_array_type
  jax_numpy_template.globals['get_axis_list'] = templates.axis_list

  # We need to make sure that the library functions we _haven't_ annotated
  # are still present in the type stubs or the type checker will think they
  # don't exist at all. We do this in a bit of a hacky way: enumerating through
  # `dir(jnp)` and adding an `Any` annotation for everything we find that's
  # not currently annotated.
  current_stubs = open(_JAX_NUMPY_STUBS_PATH).read()
  jnp_dir = []
  for x in dir(jnp):
    if (x.startswith('_')
        or f'def {x}(' in current_stubs
        or f'class {x}:' in current_stubs):
      continue
    jnp_dir.append(x)

  with open(_JAX_NUMPY_STUBS_PATH, 'w') as f:
    f.write(jax_numpy_template.render(jnp_dir=jnp_dir))


if __name__ == '__main__':
  app.run(main)

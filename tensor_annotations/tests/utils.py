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
"""Test helpers."""

import importlib
import inspect
import os
import pathlib
import re
import subprocess
import tempfile
import textwrap
import types
from typing import List
from typing import Optional


def run_pytype(code: str, check: bool) -> subprocess.CompletedProcess:  # pylint: disable=g-doc-args
  """Runs pytype on the specified code.

  Raises:
    subprocess.CalledProcessError if check=True and pytype return is non-zero

  Returns:
    A subprocess.CompletedProcess instance containing stdout
  """

  with tempfile.TemporaryDirectory() as tmp_dir:
    code_filename = os.path.join(tmp_dir, 'test.py')
    with open(code_filename, 'w') as f:
      f.write(code)

    pytype_path = pathlib.Path('pytype-single')
    tensor_annotations_dir = pathlib.Path(__file__).parent.parent
    stubs_dir = pathlib.Path(tmp_dir)
    _link_stubs(tensor_annotations_dir, stubs_dir)
    _generate_tensor_annotations_stubs(pytype_path, tensor_annotations_dir,
                                       stubs_dir)

    cmd = ([str(pytype_path), '--pythonpath', str(stubs_dir), code_filename])

    if 'TENSOR_ANNOTATIONS_DEBUG' in os.environ:
      input(f'About to run:\n{" ".join(cmd)}\nPress enter to continue: ')

    proc = subprocess.run(
        [str(pytype_path), '--pythonpath',
         str(stubs_dir), code_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,  # Raise error if non-zero return code.
    )

    if 'TENSOR_ANNOTATIONS_DEBUG' in os.environ:
      print('pytype stdout:')
      print(proc.stdout.decode())
      print('\npytype stderr:')
      print(proc.stderr.decode())

    return proc


def _link_stubs(tensor_annotations_dir: pathlib.Path, stubs_dir: pathlib.Path):
  """Link JAX/TensorFlow stubs to a place where pytype can find them."""
  google_internal = False
  if not google_internal:
    jax_module = importlib.import_module('jax-stubs')
    jax_stubs_dir = pathlib.Path(jax_module.__path__[0])
    tf_module = importlib.import_module('tensorflow-stubs')
    tf_stubs_dir = pathlib.Path(tf_module.__path__[0])
    np_module = importlib.import_module('numpy-stubs')
    np_stubs_dir = pathlib.Path(np_module.__path__[0])

  for source, target in [
      # Library functions, e.g. tf.reduce_sum.
      (jax_stubs_dir, stubs_dir / 'jax'),
      (tf_stubs_dir, stubs_dir / 'tensorflow'),
      (np_stubs_dir, stubs_dir / 'numpy'),
      # Tensor functions, e.g. Tensor.__add__.
      (tensor_annotations_dir / 'jax.pyi',
       stubs_dir / 'tensor_annotations' / 'jax.pyi'),
      (tensor_annotations_dir / 'tensorflow.pyi',
       stubs_dir / 'tensor_annotations' / 'tensorflow.pyi'),
      (tensor_annotations_dir / 'numpy.pyi',
       stubs_dir / 'tensor_annotations' / 'numpy.pyi')
  ]:
    if not os.path.exists(source):
      raise Exception(f"Stub file '{source}' does not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def _generate_tensor_annotations_stubs(pytype_path: pathlib.Path,
                                       tensor_annotations_dir: pathlib.Path,
                                       stubs_dir: pathlib.Path):
  """Generates stubs for tensor_annotations modules."""
  path = tensor_annotations_dir / 'axes.py'
  pyi_path = stubs_dir / 'tensor_annotations' / 'axes.pyi'
  pyi_path.parent.mkdir(parents=True, exist_ok=True)
  subprocess.run(
      [
          str(pytype_path),
          '-o',
          str(pyi_path),
          str(path),
      ],
      check=True,  # Raise error if non-zero return code.
  )


def assert_pytype_succeeds(code: str):
  try:
    process = run_pytype(code, check=True)
  except subprocess.CalledProcessError as e:
    print(e.stdout.decode())
    print(e.stderr.decode())
    raise e


def assert_pytype_fails(code: str):
  try:
    run_pytype(code, check=True)
  except subprocess.CalledProcessError:
    pass
  else:
    assert False, 'pytype did not raise error'


def pytype_infer_types(code: str) -> types.SimpleNamespace:
  """Runs pytype on `code`, returning inferred type of each variable.

  Args:
    code: Code to run pytype on. Should include assignment to one or more
          variables, e.g. 'x = jnp.zeros((3, 1))'. We collect a list of
          variables assigned to, and use pytype to infer type at the end of the
          code.

  Returns:
    A SimpleNamespace whose attributes map from variable names to types.

  Raises:
    Exception: If types could not be parsed from pytype output.

  For example, if `code` is
    x = 'foo'
    x = 2
    y = 3.0
  we return a SimpleNamespace `a` such that
    a.x = int
    a.y = float
  """
  # This may contain duplicates, but that's fine.
  var_names = re.findall(r'^([^: ]*).*=', code, re.MULTILINE)
  var_names = [vn.strip() for vn in var_names]  # Remove any newline prefixes
  var_names = [vn for vn in var_names if not vn.startswith('#')]  # Comments
  for var in var_names:
    code += f'\nreveal_type({var})'

  process = run_pytype(code, check=False)

  # We look at both stdout and stderr because pytype behaves differently
  # depending on whether we run the Google-internal version or the normal
  # version
  lines = (process.stdout.decode() + process.stderr.decode()).split('\n')

  return _parse_pytype_output(var_names, lines)


def _parse_pytype_output(var_names: List[str],
                         lines: List[str]) -> types.SimpleNamespace:
  """Parses the inferred type of each variable from pytype output."""
  reveal_type_lines = [l for l in lines if '[reveal-type]' in l]
  assert len(reveal_type_lines) == len(var_names)

  types_dict = {}
  for var, line in zip(var_names, reveal_type_lines):
    match = re.search(r'File "[^"]*", line \d+, in [^:]*: '
                      r'(.*) \[reveal-type\]', line)
    if match is None:
      raise Exception(f"Couldn't parse type from line: {line}")
    t = match.group(1)
    # Simplifies e.g. `tensor_annotations.jax.Array0` to just `Array0`
    t = re.sub(r'tensor_annotations.[^.]*\.', '', t)
    types_dict[var] = t

  return types.SimpleNamespace(**types_dict)


def _parse_mypy_output(var_names: List[str],
                       lines: List[str]) -> types.SimpleNamespace:
  """Parses the inferred type of each variable from Mypy output."""
  reveal_type_lines = [l for l in lines if 'Revealed type is' in l]
  assert len(reveal_type_lines) == len(var_names)

  types_dict = {}
  for var, line in zip(var_names, reveal_type_lines):
    match = re.search("Revealed type is '(.*)'", line)
    if match is None:
      raise Exception(f"Couldn't parse type from line: {line}")
    t = match.group(1)
    # Simplifies e.g. `tensor_annotations.jax.Array0` to just `Array0`
    t = re.sub(r'tensor_annotations.[^.]*\.', '', t)
    # Remove the '*' that Mypy suffixes types with if the types were inferred
    # using type variable substitution.
    t = t.replace('*', '')
    # Mypy will format axis types as e.g. `test.A1`. Get rid of the `test.`.
    t = re.sub(r'test.(A\d+)', r'\1', t)
    # Mypy will format unparameterised generics as e.g. `Tensor1[Any]`, but
    # we wrote tests assuming they'd be formatted as just `Tensor1`, so get
    # rid of the `[Any]`.
    t = t.replace('[Any]', '')
    types_dict[var] = t

  return types.SimpleNamespace(**types_dict)


def pytype_infer_shapes(
    code: str,
) -> types.SimpleNamespace:
  # pylint: disable=g-doc-args,g-doc-return-or-yield,g-doc-exception
  """Runs pytype on `code`, returning inferred shape of array/tensor variables.

  Note that shapes are inferred based on the axis labels: axis label 'A1' is
  assumed to represent a dimension of size 1, 'A2' a dimension of size 2, and so
  on. For example, we assume that a tensor of type Tensor2[int8, A1, A2] has
  shape (1, 2).

  For example, if `code` is
    x: tf.Tensor2[float32, A3, A5] = tf.zeros((3, 5))
    y = tf.transpose(x)  # tf.Tensor2[float32, A5, A3]
  we return a SimpleNamespace `a` such that
    a.x = (3, 5)
    a.y = (5, 3)

  This helper function exists so that we can easily compare the shape of the
  real outputs of shape-changing functions (e.g. tf.transpose) to the shape
  inferred by the type checker using our stubs, to confirm that our stubs
  are correct.

  See `pytype_infer_types` for more info.

  Args:
    code: The code to run pytype on.
    expect_dtype: Whether to expect a DType as the first type argument to the
      generic. This is a temporary flag used for gradually adding support for
      DTypes, and will be removed in the future.
  """
  types_namespace = pytype_infer_types(code)

  shapes_dict = {}
  var_names = [d for d in dir(types_namespace)
               if not d.startswith('_')]
  for var in var_names:
    var_type = getattr(types_namespace, var)
    if var_type == 'Any':
      shape = 'Any'
    elif 'Array' not in var_type and 'Tensor' not in var_type:
      continue
    elif var_type.endswith('Array0') or var_type.endswith('Tensor0'):
      shape = ()
    else:
      match = re.search(r'\[(.*)\]', var_type)
      if match is None:
        raise ValueError(f"Couldn't parse type '{var_type}'")
      axis_types = match.group(1)  # e.g. 'A1, A2'
      axis_types_list = axis_types.split(', ')
      unused_dtype, *shape_types = axis_types_list
      shape_str_list = [t.replace('A', '') for t in shape_types]
      shape = tuple(int(s) for s in shape_str_list)
    shapes_dict[var] = shape

  return types.SimpleNamespace(**shapes_dict)


class SaveCodeAsString:
  r"""Saves code executed within the context manager.

  Indentation is automatically adjusted such that the first line has no
  indenation in the saved code. For example, if used as follows:
    with SaveCodeString() as code_saver:
      foo = 'bar'
      f()
  then `code_saver.code` would contain "foo = 'bar'\nf()".
  """

  def __init__(self):
    self._frame_where_entered = None
    self._frame_where_exited = None
    self.code: Optional[str] = None

  def __enter__(self):
    self._frame_where_entered = inspect.stack()[1]
    return self

  def __exit__(self, *_):
    with open(self._frame_where_entered.filename, 'r') as f:
      lines = f.readlines()
    start_line_num = self._frame_where_entered.lineno - 1
    start_line = lines[start_line_num]
    start_indentation = len(start_line) - len(start_line.lstrip())

    for line_num, line in enumerate(
        lines[start_line_num + 1:],
        start=start_line_num + 1,
    ):
      if not line.strip():
        continue
      line_indentation = len(line) - len(line.lstrip())
      if line_indentation <= start_indentation:
        break
    end_line_num = line_num

    lines = lines[start_line_num + 1:end_line_num]
    self.code = ''.join(lines).rstrip()
    self.code = textwrap.dedent(self.code)

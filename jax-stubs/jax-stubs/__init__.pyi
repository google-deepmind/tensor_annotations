# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Stubs for jax.*

We also need to provide stubs for jax.Array in here to avoid breaking
code which doesn't use tensor_annotations annotations. Le sigh.

NOTE: This file is generated from templates/jax.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_jax_library_template.py
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Set
import numpy as np

Sharding = Any
Traceback = Any

import numpy as np

_sentinel: int


class Array:
  dtype: np.dtype
  ndim: int
  size: int
  itemsize: int
  aval: Any

  @property
  def shape(self) -> Tuple[int, ...]: ...

  @property
  def sharding(self) -> Sharding: ...

  @property
  def addressable_shards(self) -> Sequence[Shard]: ...

  def __init__(self, shape, dtype=None, buffer=None, offset=0, strides=None,
               order=None):
    raise TypeError("jax.numpy.ndarray() should not be instantiated explicitly."
                    " Use jax.numpy.array, or jax.numpy.zeros instead.")

  def __getitem__(self, key, indices_are_sorted=False,
                  unique_indices=False) -> Array: ...
  def __setitem__(self, key, value) -> None: ...
  def __len__(self) -> int: ...
  def __iter__(self) -> Any: ...
  def __reversed__(self) -> Any: ...
  def __round__(self, ndigits=None) -> Array: ...

  # Comparisons

  # these return bool for object, so ignore override errors.
  def __lt__(self, other) -> Array: ...  # type: ignore[override]
  def __le__(self, other) -> Array: ...  # type: ignore[override]
  def __eq__(self, other) -> Array: ...  # type: ignore[override]
  def __ne__(self, other) -> Array: ...  # type: ignore[override]
  def __gt__(self, other) -> Array: ...  # type: ignore[override]
  def __ge__(self, other) -> Array: ...  # type: ignore[override]

  # Unary arithmetic

  def __neg__(self) -> Array: ...
  def __pos__(self) -> Array: ...
  def __abs__(self) -> Array: ...
  def __invert__(self) -> Array: ...

  # Binary arithmetic

  def __add__(self, other) -> Array: ...
  def __sub__(self, other) -> Array: ...
  def __mul__(self, other) -> Array: ...
  def __matmul__(self, other) -> Array: ...
  def __truediv__(self, other) -> Array: ...
  def __floordiv__(self, other) -> Array: ...
  def __mod__(self, other) -> Array: ...
  def __divmod__(self, other) -> Array: ...
  def __pow__(self, other) -> Array: ...
  def __lshift__(self, other) -> Array: ...
  def __rshift__(self, other) -> Array: ...
  def __and__(self, other) -> Array: ...
  def __xor__(self, other) -> Array: ...
  def __or__(self, other) -> Array: ...

  def __radd__(self, other) -> Array: ...
  def __rsub__(self, other) -> Array: ...
  def __rmul__(self, other) -> Array: ...
  def __rmatmul__(self, other) -> Array: ...
  def __rtruediv__(self, other) -> Array: ...
  def __rfloordiv__(self, other) -> Array: ...
  def __rmod__(self, other) -> Array: ...
  def __rdivmod__(self, other) -> Array: ...
  def __rpow__(self, other) -> Array: ...
  def __rlshift__(self, other) -> Array: ...
  def __rrshift__(self, other) -> Array: ...
  def __rand__(self, other) -> Array: ...
  def __rxor__(self, other) -> Array: ...
  def __ror__(self, other) -> Array: ...

  def __bool__(self) -> bool: ...
  def __complex__(self) -> complex: ...
  def __int__(self) -> int: ...
  def __float__(self) -> float: ...
  def __index__(self) -> int: ...

  # np.ndarray methods:
  def all(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None) -> Array: ...
  def any(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None) -> Array: ...
  def argmax(self, axis: Optional[int] = None, out=None, keepdims=None) -> Array: ...
  def argmin(self, axis: Optional[int] = None, out=None, keepdims=None) -> Array: ...
  def argpartition(self, kth, axis=-1, kind='introselect', order=None) -> Array: ...
  def argsort(self, axis: Optional[int] = -1, kind='quicksort', order=None) -> Array: ...
  def astype(self, dtype) -> Array: ...
  def choose(self, choices, out=None, mode='raise') -> Array: ...
  def clip(self, min=None, max=None, out=None) -> Array: ...
  def compress(self, condition, axis: Optional[int] = None, out=None) -> Array: ...
  def conj(self) -> Array: ...
  def conjugate(self) -> Array: ...
  def copy(self) -> Array: ...
  def cumprod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
              dtype=None, out=None) -> Array: ...
  def cumsum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             dtype=None, out=None) -> Array: ...
  def diagonal(self, offset=0, axis1: int = 0, axis2: int = 1) -> Array: ...
  def dot(self, b, *, precision=None) -> Array: ...
  def flatten(self) -> Array: ...
  @property
  def imag(self) -> Array: ...
  def item(self, *args) -> Any: ...
  def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None, initial=None, where=None) -> Array: ...
  def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, keepdims=False, *, where=None,) -> Array: ...
  def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=None, initial=None, where=None) -> Array: ...
  @property
  def nbytes(self) -> int: ...
  def nonzero(self, *, size=None, fill_value=None) -> Array: ...
  def prod(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
           out=None, keepdims=None, initial=None, where=None) -> Array: ...
  def ptp(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
          keepdims=False,) -> Array: ...
  def ravel(self, order='C') -> Array: ...
  @property
  def real(self) -> Array: ...
  def repeat(self, repeats, axis: Optional[int] = None, *,
             total_repeat_length=None) -> Array: ...
  def reshape(self, *args, order='C') -> Array: ...
  def round(self, decimals=0, out=None) -> Array: ...
  def searchsorted(self, v, side='left', sorter=None) -> Array: ...
  def sort(self, axis: Optional[int] = -1, kind='quicksort', order=None) -> Array: ...
  def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array: ...
  def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
          dtype=None, out=None, ddof=0, keepdims=False, *, where=None) -> Array: ...
  def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype=None,
          out=None, keepdims=None, initial=None, where=None) -> Array: ...
  def swapaxes(self, axis1: int, axis2: int) -> Array: ...
  def take(self, indices, axis: Optional[int] = None, out=None,
           mode=None) -> Array: ...
  def tobytes(self, order='C') -> bytes: ...
  def tolist(self) -> List[Any]: ...
  def trace(self, offset=0, axis1: int = 0, axis2: int = 1, dtype=None,
            out=None) -> Array: ...
  def transpose(self, *args) -> Array: ...
  @property
  def T(self) -> Array: ...
  def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
          dtype=None, out=None, ddof=0, keepdims=False, *, where=None) -> Array: ...
  def view(self, dtype=None, type=None) -> Array: ...

  # Even though we don't always support the NumPy array protocol, e.g., for
  # tracer types, for type checking purposes we must declare support so we
  # implement the NumPy ArrayLike protocol.
  def __array__(self) -> np.ndarray: ...
  def __dlpack__(self) -> Any: ...

  # JAX extensions
  @property
  def at(self) -> _IndexUpdateHelper: ...
  @property
  def weak_type(self) -> bool: ...

  # Methods defined on ArrayImpl, but not on Tracers
  def addressable_data(self, index: int) -> Array: ...
  def block_until_ready(self) -> Array: ...
  def copy_to_host_async(self) -> None: ...
  def delete(self) -> None: ...
  def device(self) -> Device: ...
  def devices(self) -> Set[Device]: ...
  @property
  def global_shards(self) -> Sequence[Shard]: ...
  def is_deleted(self) -> bool: ...
  @property
  def is_fully_addressable(self) -> bool: ...
  @property
  def is_fully_replicated(self) -> bool: ...
  def on_device_size_in_bytes(self) -> int: ...
  @property
  def traceback(self) -> Traceback: ...
  def unsafe_buffer_pointer(self) -> int: ...
  @property
  def device_buffers(self) -> Any: ...


ArrayLike = Union[
  Array,  # JAX array type
  np.ndarray,  # NumPy array type
  np.bool_, np.number,  # NumPy scalar types
  bool, int, float, complex,  # Python scalar types
]


class _IndexUpdateHelper:
  def __getitem__(self, index: Any) -> _IndexUpdateRef: ...


class _IndexUpdateRef:
  def get(self, indices_are_sorted: bool = False, unique_indices: bool = False,
          mode: Optional[str] = None, fill_value: Optional[ArrayLike] = None) -> Array: ...
  def set(self, values: Any,
          indices_are_sorted: bool = False, unique_indices: bool = False,
          mode: Optional[str] = None, fill_value: Optional[ArrayLike] = None) -> Array: ...
  def add(self, values: Any, indices_are_sorted: bool = False,
          unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def mul(self, values: Any, indices_are_sorted: bool = False,
          unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def multiply(self, values: Any, indices_are_sorted: bool = False,
               unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def divide(self, values: Any, indices_are_sorted: bool = False,
             unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def power(self, values: Any, indices_are_sorted: bool = False,
            unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def min(self, values: Any, indices_are_sorted: bool = False,
          unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def max(self, values: Any, indices_are_sorted: bool = False,
          unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...
  def apply(self, func: Callable[[ArrayLike], ArrayLike], indices_are_sorted: bool = False,
            unique_indices: bool = False, mode: Optional[str] = None) -> Array: ...



Device: Any

ShapeDtypeStruct: Any

Shard: Any

__builtins__: Any

__cached__: Any

__doc__: Any

__file__: Any

__getattr__: Any

__loader__: Any

__name__: Any

__package__: Any

__path__: Any

__spec__: Any

__version__: Any

__version_info__: Any

_deprecated_ShapedArray: Any

_deprecated_ad: Any

_deprecated_curry: Any

_deprecated_flatten_fun_nokwargs: Any

_deprecated_partial_eval: Any

_deprecated_pxla: Any

_deprecated_xla: Any

_deprecations: Any

_src: Any

abstract_arrays: Any

api_util: Any

block_until_ready: Any

check_tracer_leaks: Any

checking_leaks: Any

checkpoint: Any

checkpoint_policies: Any

clear_backends: Any

clear_caches: Any

closure_convert: Any

config: Any

core: Any

custom_batching: Any

custom_derivatives: Any

custom_gradient: Any

custom_jvp: Any

custom_transpose: Any

custom_vjp: Any

debug: Any

debug_infs: Any

debug_nans: Any

default_backend: Any

default_device: Any

default_matmul_precision: Any

default_prng_impl: Any

device_count: Any

device_get: Any

device_put: Any

device_put_replicated: Any

device_put_sharded: Any

devices: Any

disable_jit: Any

distributed: Any

dtypes: Any

effects_barrier: Any

enable_checks: Any

enable_custom_prng: Any

enable_custom_vjp_by_custom_transpose: Any

ensure_compile_time_eval: Any

errors: Any

eval_shape: Any

experimental: Any

float0: Any

grad: Any

hessian: Any

host_count: Any

host_id: Any

host_ids: Any

image: Any

interpreters: Any

jacfwd: Any

jacobian: Any

jacrev: Any

jax: Any

jax2tf_associative_scan_reductions: Any

jaxlib: Any

jit: Any

jvp: Any

lax: Any

lib: Any

linear_transpose: Any

linear_util: Any

linearize: Any

live_arrays: Any

local_device_count: Any

local_devices: Any

log_compiles: Any

make_array_from_callback: Any

make_array_from_single_device_arrays: Any

make_jaxpr: Any

monitoring: Any

named_call: Any

named_scope: Any

numpy_dtype_promotion: Any

numpy_rank_promotion: Any

ops: Any

pmap: Any

print_environment_info: Any

process_count: Any

process_index: Any

profiler: Any

pure_callback: Any

random: Any

remat: Any

scipy: Any

sharding: Any

softmax_custom_jvp: Any

spmd_mode: Any

stages: Any

transfer_guard: Any

transfer_guard_device_to_device: Any

transfer_guard_device_to_host: Any

transfer_guard_host_to_device: Any

tree_flatten: Any

tree_leaves: Any

tree_map: Any

tree_structure: Any

tree_transpose: Any

tree_unflatten: Any

tree_util: Any

treedef_is_leaf: Any

typing: Any

util: Any

value_and_grad: Any

version: Any

vjp: Any

vmap: Any

xla_computation: Any

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
"""TensorFlow stubs.

NOTE: This file is generated from templates/tensorflow.pyi
      using a Google-internal tool.
"""

# BEGIN: tensor_annotations annotations
from typing import Any, TypeVar, Tuple, overload
from typing_extensions import Literal

from tensor_annotations.axes import Axis
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)
A5 = TypeVar('A5', bound=Axis)
# This alias makes the meaning clearer in code.
# Unfortunately, it still shows up as 'Any' in pytype output.
AnyDType = Any
DT = TypeVar('DT')

TRUE = Literal[True]
FALSE = Literal[False]

LN1 = Literal[-1]
L0 = Literal[0]
L1 = Literal[1]
L2 = Literal[2]
L3 = Literal[3]
L4 = Literal[4]

Shape1 = Tuple[int]
Shape2 = Tuple[int, int]
Shape3 = Tuple[int, int, int]
Shape4 = Tuple[int, int, int, int]
Shape5 = Tuple[int, int, int, int, int]
# END: tensor_annotations annotations

import enum
from typing import Any

def __getattr__(name) -> Any: ...

_HAS_DYNAMIC_ATTRIBUTES = True

class AggregationMethod(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  ADD_N: Any
  DEFAULT: Any
  EXPERIMENTAL_ACCUMULATE_N: Any
  EXPERIMENTAL_TREE: Any
  def __init__(*args, **kwargs) -> None: ...

class CriticalSection(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  name: Any
  def __init__(self, name=..., shared_name=..., critical_section_def=..., import_scope=...) -> None: ...
  def execute(self, fn, exclusive_resource_access=..., name=...) -> Any: ...

class DType():
  _HAS_DYNAMIC_ATTRIBUTES = True
  as_datatype_enum: Any
  as_numpy_dtype: Any
  base_dtype: Any
  is_bool: Any
  is_complex: Any
  is_floating: Any
  is_integer: Any
  is_numpy_compatible: Any
  is_quantized: Any
  is_unsigned: Any
  limits: Any
  max: Any
  min: Any
  name: Any
  real_dtype: Any
  size: Any
  def __init__(*args, **kwargs) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  def is_compatible_with(self, other) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, types) -> Any: ...

class DeviceSpec(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  device_index: Any
  device_type: Any
  job: Any
  replica: Any
  task: Any
  def __init__(self, job=..., replica=..., task=..., device_type=..., device_index=...) -> None: ...
  @classmethod
  def from_string(cls, spec) -> Any: ...
  def make_merged_spec(self, dev) -> Any: ...
  def parse_from_string(self, spec) -> Any: ...
  def replace(self, **kwargs) -> Any: ...
  def to_string(self) -> Any: ...

class GradientTape(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(self, persistent=..., watch_accessed_variables=...) -> None: ...
  def batch_jacobian(self, target, source, unconnected_gradients=..., parallel_iterations=..., experimental_use_pfor=...) -> Any: ...
  def gradient(self, target, sources, output_gradients=..., unconnected_gradients=...) -> Any: ...
  def jacobian(self, target, sources, unconnected_gradients=..., parallel_iterations=..., experimental_use_pfor=...) -> Any: ...
  def reset(self) -> Any: ...
  def stop_recording(self) -> Any: ...
  def watch(self, tensor) -> Any: ...
  def watched_variables(self) -> Any: ...

class Graph(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  building_function: Any
  collections: Any
  finalized: Any
  graph_def_versions: Any
  seed: Any
  version: Any
  def __init__(self) -> None: ...
  def add_to_collection(self, name, value) -> Any: ...
  def add_to_collections(self, names, value) -> Any: ...
  def as_default(self) -> Any: ...
  def as_graph_def(self, from_version=..., add_shapes=...) -> Any: ...
  def as_graph_element(self, obj, allow_tensor=..., allow_operation=...) -> Any: ...
  def clear_collection(self, name) -> Any: ...
  def colocate_with(self, op, ignore_existing=...) -> Any: ...
  def container(self, container_name) -> Any: ...
  def control_dependencies(self, control_inputs) -> Any: ...
  def create_op(self, op_type, inputs, dtypes=..., input_types=..., name=..., attrs=..., op_def=..., compute_shapes=..., compute_device=...) -> Any: ...
  def device(self, device_name_or_function) -> Any: ...
  def finalize(self) -> Any: ...
  def get_all_collection_keys(self) -> Any: ...
  def get_collection(self, name, scope=...) -> Any: ...
  def get_collection_ref(self, name) -> Any: ...
  def get_name_scope(self) -> Any: ...
  def get_operation_by_name(self, name) -> Any: ...
  def get_operations(self) -> Any: ...
  def get_tensor_by_name(self, name) -> Any: ...
  def gradient_override_map(self, op_type_map) -> Any: ...
  def is_feedable(self, tensor) -> Any: ...
  def is_fetchable(self, tensor_or_op) -> Any: ...
  def name_scope(self, name) -> Any: ...
  def prevent_feeding(self, tensor) -> Any: ...
  def prevent_fetching(self, op) -> Any: ...
  def switch_to_thread_local(self) -> Any: ...
  def unique_name(self, name, mark_as_used=...) -> Any: ...

class IndexedSlices(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dense_shape: Any
  device: Any
  dtype: Any
  graph: Any
  indices: Any
  name: Any
  op: Any
  shape: Any
  values: Any
  def __init__(self, values, indices, dense_shape=...) -> None: ...
  def consumers(self) -> Any: ...

class IndexedSlicesSpec(TypeSpec, object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  value_type: Any
  def __init__(self, shape=..., dtype=..., indices_dtype=..., dense_shape_dtype=..., indices_shape=...) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  def is_compatible_with(self, spec_or_value) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class Module(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  name: Any
  name_scope: Any
  non_trainable_variables: Any
  submodules: Any
  trainable_variables: Any
  variables: Any
  def __init__(self, name=...) -> None: ...
  @classmethod
  def with_name_scope(cls, method) -> Any: ...

class Operation(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  control_inputs: Any
  device: Any
  graph: Any
  inputs: Any
  name: Any
  node_def: Any
  op_def: Any
  outputs: Any
  traceback: Any
  type: Any
  def __init__(self, node_def, g, inputs=..., output_types=..., control_inputs=..., input_types=..., original_op=..., op_def=...) -> None: ...
  def colocation_groups(self) -> Any: ...
  def experimental_set_type(self, type_proto) -> Any: ...
  def get_attr(self, name) -> Any: ...
  def run(self, feed_dict=..., session=...) -> Any: ...
  def values(self) -> Any: ...

class OptionalSpec(TypeSpec, object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  value_type: Any
  def __init__(self, element_spec) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  def from_value(value) -> Any: ...
  def is_compatible_with(self, spec_or_value) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class RaggedTensor(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dtype: Any
  flat_values: Any
  nested_row_splits: Any
  ragged_rank: Any
  row_splits: Any
  shape: Any
  uniform_row_length: Any
  values: Any
  def __init__(self, values, row_partition, internal=...) -> None: ...
  def bounding_shape(self, axis=..., name=..., out_type=...) -> Any: ...
  def consumers(self) -> Any: ...
  @classmethod
  def from_nested_row_lengths(cls, flat_values, nested_row_lengths, name=..., validate=...) -> Any: ...
  @classmethod
  def from_nested_row_splits(cls, flat_values, nested_row_splits, name=..., validate=...) -> Any: ...
  @classmethod
  def from_nested_value_rowids(cls, flat_values, nested_value_rowids, nested_nrows=..., name=..., validate=...) -> Any: ...
  @classmethod
  def from_row_lengths(cls, values, row_lengths, name=..., validate=...) -> Any: ...
  @classmethod
  def from_row_limits(cls, values, row_limits, name=..., validate=...) -> Any: ...
  @classmethod
  def from_row_splits(cls, values, row_splits, name=..., validate=...) -> Any: ...
  @classmethod
  def from_row_starts(cls, values, row_starts, name=..., validate=...) -> Any: ...
  @classmethod
  def from_sparse(cls, st_input, name=..., row_splits_dtype=...) -> Any: ...
  @classmethod
  def from_tensor(cls, tensor, lengths=..., padding=..., ragged_rank=..., name=..., row_splits_dtype=...) -> Any: ...
  @classmethod
  def from_uniform_row_length(cls, values, uniform_row_length, nrows=..., validate=..., name=...) -> Any: ...
  @classmethod
  def from_value_rowids(cls, values, value_rowids, nrows=..., name=..., validate=...) -> Any: ...
  def get_shape(self) -> Any: ...
  def merge_dims(self, outer_axis, inner_axis) -> Any: ...
  def nested_row_lengths(self, name=...) -> Any: ...
  def nested_value_rowids(self, name=...) -> Any: ...
  def nrows(self, out_type=..., name=...) -> Any: ...
  def numpy(self) -> Any: ...
  def row_lengths(self, axis=..., name=...) -> Any: ...
  def row_limits(self, name=...) -> Any: ...
  def row_starts(self, name=...) -> Any: ...
  def to_list(self) -> Any: ...
  def to_sparse(self, name=...) -> Any: ...
  def to_tensor(self, default_value=..., name=..., shape=...) -> Any: ...
  def value_rowids(self, name=...) -> Any: ...
  def with_flat_values(self, new_values) -> Any: ...
  def with_row_splits_dtype(self, dtype) -> Any: ...
  def with_values(self, new_values) -> Any: ...

class RaggedTensorSpec(TypeSpec, object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dtype: Any
  flat_values_spec: Any
  ragged_rank: Any
  row_splits_dtype: Any
  shape: Any
  value_type: Any
  def __init__(self, shape=..., dtype=..., ragged_rank=..., row_splits_dtype=..., flat_values_spec=...) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  @classmethod
  def from_value(cls, value) -> Any: ...
  def is_compatible_with(self, spec_or_value) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class RegisterGradient(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(self, op_type) -> None: ...

class SparseTensor(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dense_shape: Any
  dtype: Any
  graph: Any
  indices: Any
  op: Any
  shape: Any
  values: Any
  def __init__(self, indices, values, dense_shape) -> None: ...
  def consumers(self) -> Any: ...
  def eval(self, feed_dict=..., session=...) -> Any: ...
  @classmethod
  def from_value(cls, sparse_tensor_value) -> Any: ...
  def get_shape(self) -> Any: ...
  def with_values(self, new_values) -> Any: ...

class SparseTensorSpec(TypeSpec, object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dtype: Any
  shape: Any
  value_type: Any
  def __init__(self, shape=..., dtype=...) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  @classmethod
  def from_value(cls, value) -> Any: ...
  def is_compatible_with(self, spec_or_value) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class Summary(object):
  def __init__(self, value=...) -> None: ...
  def Value(**kwargs) -> Any: ...

class Tensor(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  OVERLOADABLE_OPERATORS: Any
  device: Any
  dtype: Any
  graph: Any
  name: Any
  op: Any
  shape: Any
  value_index: Any
  def __init__(self, op, value_index, dtype) -> None: ...
  def consumers(self) -> Any: ...
  def eval(self, feed_dict=..., session=...) -> Any: ...
  def experimental_ref(self) -> Any: ...
  def get_shape(self) -> Any: ...
  def ref(self) -> Any: ...
  def set_shape(self, shape) -> Any: ...

class TensorArray(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dtype: Any
  dynamic_size: Any
  element_shape: Any
  flow: Any
  handle: Any
  def __init__(self, dtype, size=..., dynamic_size=..., clear_after_read=..., tensor_array_name=..., handle=..., flow=..., infer_shape=..., element_shape=..., colocate_with_first_write_call=..., name=...) -> None: ...
  def close(self, name=...) -> Any: ...
  def concat(self, name=...) -> Any: ...
  def gather(self, indices, name=...) -> Any: ...
  def grad(self, source, flow=..., name=...) -> Any: ...
  def identity(self) -> Any: ...
  def read(self, index, name=...) -> Any: ...
  def scatter(self, indices, value, name=...) -> Any: ...
  def size(self, name=...) -> Any: ...
  def split(self, value, lengths, name=...) -> Any: ...
  def stack(self, name=...) -> Any: ...
  def unstack(self, value, name=...) -> Any: ...
  def write(self, index, value, name=...) -> Any: ...

class TensorArraySpec(TypeSpec, object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  value_type: Any
  def __init__(self, element_shape=..., dtype=..., dynamic_size=..., infer_shape=...) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  def from_value(value) -> Any: ...
  def is_compatible_with(self, other) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class TensorShape(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dims: Any
  ndims: Any
  rank: Any
  def __init__(self, dims) -> None: ...
  def as_list(self) -> list[int]: ...
  def as_proto(self) -> Any: ...
  def assert_has_rank(self, rank) -> Any: ...
  def assert_is_compatible_with(self, other) -> Any: ...
  def assert_is_fully_defined(self) -> Any: ...
  def assert_same_rank(self, other) -> Any: ...
  def concatenate(self, other) -> Any: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  def is_compatible_with(self, other) -> Any: ...
  def is_fully_defined(self) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def merge_with(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_shape(self, other) -> Any: ...
  def num_elements(self) -> Any: ...
  def with_rank(self, rank) -> Any: ...
  def with_rank_at_least(self, rank) -> Any: ...
  def with_rank_at_most(self, rank) -> Any: ...

  @overload
  def __getitem__(self, key: int) -> int: ...

  @overload
  def __getitem__(self, key: slice) -> TensorShape: ...

class TensorSpec(TypeSpec, object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  dtype: Any
  name: Any
  shape: Any
  value_type: Any
  def __init__(self, shape, dtype=..., name=...) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  @classmethod
  def from_spec(cls, spec, name=...) -> Any: ...
  @classmethod
  def from_tensor(cls, tensor, name=...) -> Any: ...
  def is_compatible_with(self, spec_or_tensor) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class TypeSpec(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  value_type: Any
  def __init__(*args, **kwargs) -> None: ...
  def experimental_as_proto(self) -> Any: ...
  @classmethod
  def experimental_from_proto(cls, proto) -> Any: ...
  @classmethod
  def experimental_type_proto(cls) -> Any: ...
  def is_compatible_with(self, spec_or_value) -> Any: ...
  def is_subtype_of(self, other) -> Any: ...
  def most_specific_common_supertype(self, others) -> Any: ...
  def most_specific_compatible_type(self, other) -> Any: ...

class UnconnectedGradients(enum.Enum):
  _HAS_DYNAMIC_ATTRIBUTES = True
  NONE: Any
  ZERO: Any

class Variable(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  aggregation: Any
  constraint: Any
  device: Any
  dtype: Any
  graph: Any
  initial_value: Any
  initializer: Any
  name: Any
  op: Any
  shape: Any
  synchronization: Any
  trainable: Any
  def __init__(self, initial_value=..., trainable=..., validate_shape=..., caching_device=..., name=..., variable_def=..., dtype=..., import_scope=..., constraint=..., synchronization=..., aggregation=..., shape=..., experimental_enable_variable_lifting=...) -> None: ...
  def assign(self, value, use_locking=..., name=..., read_value=...) -> Any: ...
  def assign_add(self, delta, use_locking=..., name=..., read_value=...) -> Any: ...
  def assign_sub(self, delta, use_locking=..., name=..., read_value=...) -> Any: ...
  def batch_scatter_update(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def count_up_to(self, limit) -> Any: ...
  def eval(self, session=...) -> Any: ...
  def experimental_ref(self) -> Any: ...
  def from_proto(variable_def, import_scope=...) -> Any: ...
  def gather_nd(self, indices, name=...) -> Any: ...
  def get_shape(self) -> Any: ...
  def initialized_value(self) -> Any: ...
  def load(self, value, session=...) -> Any: ...
  def read_value(self) -> Any: ...
  def ref(self) -> Any: ...
  def scatter_add(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def scatter_div(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def scatter_max(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def scatter_min(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def scatter_mul(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def scatter_nd_add(self, indices, updates, name=...) -> Any: ...
  def scatter_nd_sub(self, indices, updates, name=...) -> Any: ...
  def scatter_nd_update(self, indices, updates, name=...) -> Any: ...
  def scatter_sub(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def scatter_update(self, sparse_delta, use_locking=..., name=...) -> Any: ...
  def set_shape(self, shape) -> Any: ...
  def sparse_read(self, indices, name=...) -> Any: ...
  def to_proto(self, export_scope=...) -> Any: ...
  def value(self) -> Any: ...

class VariableAggregation(enum.Enum):
  _HAS_DYNAMIC_ATTRIBUTES = True
  MEAN: Any
  NONE: Any
  ONLY_FIRST_REPLICA: Any
  SUM: Any

class VariableSynchronization(enum.Enum):
  _HAS_DYNAMIC_ATTRIBUTES = True
  AUTO: Any
  NONE: Any
  ON_READ: Any
  ON_WRITE: Any

bfloat16: Any
bool: Any
complex128: Any
complex64: Any
class constant_initializer(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(self, value=...) -> None: ...
  @classmethod
  def from_config(cls, config) -> Any: ...
  def get_config(self) -> Any: ...

double: Any
float16: Any
float32: Any
float64: Any
half: Any
int16: Any
int32: Any
int64: Any
int8: Any
class name_scope(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  name: Any
  def __init__(self, name) -> None: ...

newaxis: Any
class ones_initializer(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(*args, **kwargs) -> None: ...
  @classmethod
  def from_config(cls, config) -> Any: ...
  def get_config(self) -> Any: ...

qint16: Any
qint32: Any
qint8: Any
quint16: Any
quint8: Any
class random_normal_initializer(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(self, mean=..., stddev=..., seed=...) -> None: ...
  @classmethod
  def from_config(cls, config) -> Any: ...
  def get_config(self) -> Any: ...

class random_uniform_initializer(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(self, minval=..., maxval=..., seed=...) -> None: ...
  @classmethod
  def from_config(cls, config) -> Any: ...
  def get_config(self) -> Any: ...

resource: Any
string: Any
uint16: Any
uint32: Any
uint64: Any
uint8: Any
variant: Any
class zeros_initializer(object):
  _HAS_DYNAMIC_ATTRIBUTES = True
  def __init__(*args, **kwargs) -> None: ...
  @classmethod
  def from_config(cls, config) -> Any: ...
  def get_config(self) -> Any: ...

def Assert(condition, data, summarize=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for abs(...)
@overload
def abs(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def abs(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def abs(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def abs(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def abs(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def abs(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def abs(x, name=...) -> Any: ...
# END: tensor_annotations annotations for abs(...)


# BEGIN: tensor_annotations annotations for acos(...)
@overload
def acos(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def acos(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def acos(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def acos(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def acos(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def acos(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def acos(x, name=...) -> Any: ...
# END: tensor_annotations annotations for acos(...)


# BEGIN: tensor_annotations annotations for acosh(...)
@overload
def acosh(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def acosh(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def acosh(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def acosh(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def acosh(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def acosh(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def acosh(x, name=...) -> Any: ...
# END: tensor_annotations annotations for acosh(...)

def add(x, y, name=...) -> Any: ...
def add_n(inputs, name=...) -> Any: ...
def approx_top_k(input, k, reduction_dimension=..., recall_target=..., is_max_k=..., reduction_input_size_override=..., aggregate_to_topk=..., name=...) -> Any: ...
def argmax(input, axis=..., output_type=..., name=...) -> Any: ...
def argmin(input, axis=..., output_type=..., name=...) -> Any: ...
def argsort(values, axis=..., direction=..., stable=..., name=...) -> Any: ...
def as_dtype(type_value) -> Any: ...
def as_string(input, precision=..., scientific=..., shortest=..., width=..., fill=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for asin(...)
@overload
def asin(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def asin(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def asin(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def asin(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def asin(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def asin(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def asin(x, name=...) -> Any: ...
# END: tensor_annotations annotations for asin(...)


# BEGIN: tensor_annotations annotations for asinh(...)
@overload
def asinh(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def asinh(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def asinh(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def asinh(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def asinh(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def asinh(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def asinh(x, name=...) -> Any: ...
# END: tensor_annotations annotations for asinh(...)

def assert_equal(x, y, message=..., summarize=..., name=...) -> Any: ...
def assert_greater(x, y, message=..., summarize=..., name=...) -> Any: ...
def assert_less(x, y, message=..., summarize=..., name=...) -> Any: ...
def assert_rank(x, rank, message=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for atan(...)
@overload
def atan(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def atan(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def atan(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def atan(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def atan(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def atan(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def atan(x, name=...) -> Any: ...
# END: tensor_annotations annotations for atan(...)

def atan2(y, x, name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for atanh(...)
@overload
def atanh(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def atanh(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def atanh(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def atanh(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def atanh(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def atanh(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def atanh(x, name=...) -> Any: ...
# END: tensor_annotations annotations for atanh(...)

def batch_to_space(input, block_shape, crops, name=...) -> Any: ...
def bitcast(input, type, name=...) -> Any: ...
def boolean_mask(tensor, mask, axis=..., name=...) -> Any: ...
def broadcast_dynamic_shape(shape_x, shape_y) -> Any: ...
def broadcast_static_shape(shape_x, shape_y) -> Any: ...
def broadcast_to(input, shape, name=...) -> Any: ...
def case(pred_fn_pairs, default=..., exclusive=..., strict=..., name=...) -> Any: ...
def cast(x, dtype, name=...) -> Any: ...
def clip_by_global_norm(t_list, clip_norm, use_norm=..., name=...) -> Any: ...
def clip_by_norm(t, clip_norm, axes=..., name=...) -> Any: ...
def clip_by_value(t, clip_value_min, clip_value_max, name=...) -> Any: ...
def complex(real, imag, name=...) -> Any: ...
def concat(values, axis, name=...) -> Any: ...
def cond(pred, true_fn=..., false_fn=..., name=...) -> Any: ...
def constant(value, dtype=..., shape=..., name=...) -> Any: ...
def control_dependencies(control_inputs) -> Any: ...
def convert_to_tensor(value, dtype=..., dtype_hint=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for cos(...)
@overload
def cos(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def cos(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def cos(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def cos(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def cos(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def cos(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def cos(x, name=...) -> Any: ...
# END: tensor_annotations annotations for cos(...)


# BEGIN: tensor_annotations annotations for cosh(...)
@overload
def cosh(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def cosh(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def cosh(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def cosh(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def cosh(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def cosh(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def cosh(x, name=...) -> Any: ...
# END: tensor_annotations annotations for cosh(...)

def cumsum(x, axis=..., exclusive=..., reverse=..., name=...) -> Any: ...
def custom_gradient(f=...) -> Any: ...
def device(device_name) -> Any: ...
def divide(x, y, name=...) -> Any: ...
def dynamic_partition(data, partitions, num_partitions, name=...) -> Any: ...
def dynamic_stitch(indices, data, name=...) -> Any: ...
def edit_distance(hypothesis, truth, normalize=..., name=...) -> Any: ...
def eig(tensor, name=...) -> Any: ...
def eigvals(tensor, name=...) -> Any: ...
def einsum(equation, *inputs, **kwargs) -> Any: ...
def ensure_shape(x, shape, name=...) -> Any: ...
def equal(x, y, name=...) -> Any: ...
def executing_eagerly() -> Any: ...

# BEGIN: tensor_annotations annotations for exp(...)
@overload
def exp(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def exp(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def exp(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def exp(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def exp(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def exp(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def exp(x, name=...) -> Any: ...
# END: tensor_annotations annotations for exp(...)

def expand_dims(input, axis, name=...) -> Any: ...
def extract_volume_patches(input, ksizes, strides, padding, name=...) -> Any: ...
def eye(num_rows, num_columns=..., batch_shape=..., dtype=..., name=...) -> Any: ...
def fill(dims, value, name=...) -> Any: ...
def fingerprint(data, method=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for floor(...)
@overload
def floor(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def floor(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def floor(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def floor(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def floor(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def floor(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def floor(x, name=...) -> Any: ...
# END: tensor_annotations annotations for floor(...)

def foldl(fn, elems, initializer=..., parallel_iterations=..., back_prop=..., swap_memory=..., name=...) -> Any: ...
def foldr(fn, elems, initializer=..., parallel_iterations=..., back_prop=..., swap_memory=..., name=...) -> Any: ...
def function(func=..., input_signature=..., autograph=..., jit_compile=..., reduce_retracing=..., experimental_implements=..., experimental_autograph_options=..., experimental_relax_shapes=..., experimental_compile=..., experimental_follow_type_hints=...) -> Any: ...
def gather(params, indices, validate_indices=..., axis=..., batch_dims=..., name=...) -> Any: ...
def gather_nd(params, indices, batch_dims=..., name=...) -> Any: ...
def get_current_name_scope() -> Any: ...
def get_logger() -> Any: ...
def get_static_value(tensor, partial=...) -> Any: ...
def get_variable(name, shape=..., dtype=..., initializer=..., regularizer=..., trainable=..., collections=..., caching_device=..., partitioner=..., validate_shape=..., use_resource=..., custom_getter=..., constraint=..., synchronization=..., aggregation=...) -> Any: ...
def grad_pass_through(f) -> Any: ...
def gradients(ys, xs, grad_ys=..., name=..., gate_gradients=..., aggregation_method=..., stop_gradients=..., unconnected_gradients=...) -> Any: ...
def greater(x, y, name=...) -> Any: ...
def greater_equal(x, y, name=...) -> Any: ...
def group(*inputs, **kwargs) -> Any: ...
def guarantee_const(input, name=...) -> Any: ...
def hessians(ys, xs, gate_gradients=..., aggregation_method=..., name=...) -> Any: ...
def histogram_fixed_width(values, value_range, nbins=..., dtype=..., name=...) -> Any: ...
def histogram_fixed_width_bins(values, value_range, nbins=..., dtype=..., name=...) -> Any: ...
def identity(input, name=...) -> Any: ...
def identity_n(input, name=...) -> Any: ...
def import_graph_def(graph_def, input_map=..., return_elements=..., name=..., op_dict=..., producer_op_list=...) -> Any: ...
def init_scope() -> Any: ...
def inside_function() -> Any: ...
def is_tensor(x) -> Any: ...
def less(x, y, name=...) -> Any: ...
def less_equal(x, y, name=...) -> Any: ...
def linspace(start, stop, num, name=..., axis=...) -> Any: ...
def load_library(library_location) -> Any: ...
def load_op_library(library_filename) -> Any: ...
def logical_and(x, y, name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for logical_not(...)
@overload
def logical_not(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def logical_not(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def logical_not(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def logical_not(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def logical_not(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def logical_not(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def logical_not(x, name=...) -> Any: ...
# END: tensor_annotations annotations for logical_not(...)

def logical_or(x, y, name=...) -> Any: ...
def make_ndarray(tensor) -> Any: ...
def make_tensor_proto(values, dtype=..., shape=..., verify_shape=..., allow_broadcast=...) -> Any: ...
def map_fn(fn, elems, dtype=..., parallel_iterations=..., back_prop=..., swap_memory=..., infer_shape=..., name=..., fn_output_signature=...) -> Any: ...

# BEGIN: tensor_annotations annotations for matmul(...)
@overload
def matmul(
    a: Tensor2[AnyDType, A1, A2],
    b: Tensor2[AnyDType, A2, A3],
    name=...,
) -> Tensor2[AnyDType, A1, A3]: ...

@overload
def matmul(
    a: Tensor2[AnyDType, A1, A2],
    b: Tensor2[AnyDType, A1, A3],
    transpose_a: TRUE,
    name=...
) -> Tensor2[AnyDType, A2, A3]: ...

@overload
def matmul(
    a: Tensor2[AnyDType, A1, A2],
    b: Tensor2[AnyDType, A3, A2],
    transpose_b: TRUE,
    name=...
) -> Tensor2[AnyDType, A1, A3]: ...

@overload
def matmul(
    a: Tensor3[AnyDType, A1, A2, A3],
    b: Tensor2[AnyDType, A3, A4],
    name=...
) -> Tensor3[AnyDType, A1, A2, A4]: ...

@overload
def matmul(
    a: Tensor3[AnyDType, A1, A2, A3],
    b: Tensor2[AnyDType, A4, A3],
    transpose_b: TRUE,
    name=...
) -> Tensor3[AnyDType, A4, A2, A3]: ...

@overload
def matmul(
    a, b,
    transpose_a=..., transpose_b=...,
    adjoint_a=..., adjoint_b=...,
    a_is_sparse=..., b_is_sparse=...,
    name=...
) -> Any: ...
# END: tensor_annotations annotations for matmul(...)

def matrix_square_root(input, name=...) -> Any: ...
def maximum(x, y, name=...) -> Any: ...
def meshgrid(*args, **kwargs) -> Any: ...
def minimum(x, y, name=...) -> Any: ...
def multiply(x, y, name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for negative(...)
@overload
def negative(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def negative(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def negative(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def negative(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def negative(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def negative(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def negative(x, name=...) -> Any: ...
# END: tensor_annotations annotations for negative(...)

def no_gradient(op_type) -> Any: ...
def no_op(name=...) -> Any: ...
def nondifferentiable_batch_function(num_batch_threads, max_batch_size, batch_timeout_micros, allowed_batch_sizes=..., max_enqueued_batches=..., autograph=..., enable_large_batch_splitting=...) -> Any: ...
def norm(tensor, ord=..., axis=..., keepdims=..., name=...) -> Any: ...
def not_equal(x, y, name=...) -> Any: ...
def numpy_function(func, inp, Tout, stateful=..., name=...) -> Any: ...
def one_hot(indices, depth, on_value=..., off_value=..., axis=..., dtype=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for ones(...)
@overload
def ones(shape: Shape1, dtype=..., name=...) -> Tensor1[AnyDType, Any]: ...

@overload
def ones(shape: Shape2, dtype=..., name=...) -> Tensor2[AnyDType, Any, Any]: ...

@overload
def ones(shape: Shape3, dtype=..., name=...) -> Tensor3[AnyDType, Any, Any, Any]: ...

@overload
def ones(shape: Shape4, dtype=..., name=...) -> Tensor4[AnyDType, Any, Any, Any, Any]: ...

@overload
def ones(shape: Shape5, dtype=..., name=...) -> Tensor5[AnyDType, Any, Any, Any, Any, Any]: ...

# See note about Tensor0 in `zeros`
@overload
def ones(shape: Tuple[()], dtype=..., name=...) -> Tensor0[AnyDType]: ...

@overload
def ones(shape, dtype=..., name=...) -> AnyDType: ...
# END: tensor_annotations annotations for ones(...)


# BEGIN: tensor_annotations annotations for ones_like(...)
@overload
def ones_like(input: Tensor1[AnyDType, A1], dtype=..., name=...) -> Tensor1[AnyDType, A1]: ...

@overload
def ones_like(input: Tensor2[AnyDType, A1, A2], dtype=..., name=...) -> Tensor2[AnyDType, A1, A2]: ...

@overload
def ones_like(input: Tensor3[AnyDType, A1, A2, A3], dtype=..., name=...) -> Tensor3[AnyDType, A1, A2, A3]: ...

@overload
def ones_like(input: Tensor4[AnyDType, A1, A2, A3, A4], dtype=..., name=...) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...

@overload
def ones_like(input: Tensor5[AnyDType, A1, A2, A3, A4, A5], dtype=..., name=...) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...

@overload
def ones_like(input, dtype=..., name=...) -> Any: ...
# END: tensor_annotations annotations for ones_like(...)

def pad(tensor, paddings, mode=..., constant_values=..., name=...) -> Any: ...
def parallel_stack(values, name=...) -> Any: ...
def placeholder(dtype, shape=..., name=...) -> Any: ...
def pow(x, y, name=...) -> Any: ...
def print(*inputs, **kwargs) -> Any: ...
def py_function(func, inp, Tout, name=...) -> Any: ...
def random_index_shuffle(index, seed, max_index, rounds=..., name=...) -> Any: ...
def range(start, limit=..., delta=..., dtype=..., name=...) -> Any: ...
def rank(input, name=...) -> Any: ...
def realdiv(x, y, name=...) -> Any: ...
def recompute_grad(f) -> Any: ...

# BEGIN: tensor_annotations annotations for reduce_all(...)
@overload
def reduce_all(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_all(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_all(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_all(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_all(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_all(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_all(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_all(...)


# BEGIN: tensor_annotations annotations for reduce_any(...)
@overload
def reduce_any(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_any(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_any(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_any(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_any(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_any(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_any(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_any(...)


# BEGIN: tensor_annotations annotations for reduce_logsumexp(...)
@overload
def reduce_logsumexp(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_logsumexp(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_logsumexp(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_logsumexp(...)


# BEGIN: tensor_annotations annotations for reduce_max(...)
@overload
def reduce_max(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_max(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_max(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_max(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_max(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_max(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_max(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_max(...)


# BEGIN: tensor_annotations annotations for reduce_mean(...)
@overload
def reduce_mean(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_mean(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_mean(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_mean(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_mean(...)


# BEGIN: tensor_annotations annotations for reduce_min(...)
@overload
def reduce_min(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_min(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_min(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_min(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_min(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_min(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_min(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_min(...)


# BEGIN: tensor_annotations annotations for reduce_prod(...)
@overload
def reduce_prod(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_prod(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_prod(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_prod(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_prod(...)


# BEGIN: tensor_annotations annotations for reduce_sum(...)
@overload
def reduce_sum(input_tensor: Tensor1[DT, A1],
               axis: L0, name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor1[DT, A1],
               axis: LN1, name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor2[DT, A1, A2],
               axis: L0, name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor2[DT, A1, A2],
               axis: L1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor2[DT, A1, A2],
               axis: LN1, name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, L1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor2[DT, A1, A2],
               axis: Tuple[L0, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L0, name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L1, name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: L2, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: LN1, name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L2], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, L2], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L1, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A1, A2, A3],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L0, name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L1, name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L2, name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: L3, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: LN1, name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L3], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L3], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, L3], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L2, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A1, A2, A3, A4],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L0, name=...) -> Tensor4[DT, A2, A3, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L1, name=...) -> Tensor4[DT, A1, A3, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L2, name=...) -> Tensor4[DT, A1, A2, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L3, name=...) -> Tensor4[DT, A1, A2, A3, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: L4, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: LN1, name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1], name=...) -> Tensor3[DT, A3, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2], name=...) -> Tensor3[DT, A2, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3], name=...) -> Tensor3[DT, A2, A3, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L4], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, LN1], name=...) -> Tensor3[DT, A2, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2], name=...) -> Tensor3[DT, A1, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3], name=...) -> Tensor3[DT, A1, A3, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L4], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, LN1], name=...) -> Tensor3[DT, A1, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3], name=...) -> Tensor3[DT, A1, A2, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L4], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, LN1], name=...) -> Tensor3[DT, A1, A2, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, L4], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L3, LN1], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2], name=...) -> Tensor2[DT, A4, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3], name=...) -> Tensor2[DT, A3, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L4], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, LN1], name=...) -> Tensor2[DT, A3, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3], name=...) -> Tensor2[DT, A2, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L4], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, LN1], name=...) -> Tensor2[DT, A2, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, L4], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L3, LN1], name=...) -> Tensor2[DT, A2, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3], name=...) -> Tensor2[DT, A1, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L4], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, LN1], name=...) -> Tensor2[DT, A1, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, L4], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L3, LN1], name=...) -> Tensor2[DT, A1, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, L4], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L2, L3, LN1], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3], name=...) -> Tensor1[DT, A5]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L4], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, LN1], name=...) -> Tensor1[DT, A4]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, L4], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L3, LN1], name=...) -> Tensor1[DT, A3]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, L4], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L2, L3, LN1], name=...) -> Tensor1[DT, A2]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, L4], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L1, L2, L3, LN1], name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, L4], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A1, A2, A3, A4, A5],
               axis: Tuple[L0, L1, L2, L3, LN1], name=...) -> Tensor0[DT]: ...

@overload
def reduce_sum(input_tensor: Tensor1[DT, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor1[DT, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor2[DT, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor3[DT, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor4[DT, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def reduce_sum(input_tensor: Tensor5[DT, A5, A4, A3, A2, A1],
               axis=..., keepdims: TRUE = ..., name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def reduce_sum(input_tensor, axis=..., keepdims=..., name=...) -> Any: ...
# END: tensor_annotations annotations for reduce_sum(...)

def register_tensor_conversion_function(base_type, conversion_func, priority=...) -> Any: ...
def repeat(input, repeats, axis=..., name=...) -> Any: ...
def required_space_to_batch_paddings(input_shape, block_shape, base_paddings=..., name=...) -> Any: ...
def reshape(tensor, shape, name=...) -> Any: ...
def reverse(tensor, axis, name=...) -> Any: ...
def reverse_sequence(input, seq_lengths, seq_axis=..., batch_axis=..., name=...) -> Any: ...
def roll(input, shift, axis, name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for round(...)
@overload
def round(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def round(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def round(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def round(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def round(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def round(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def round(x, name=...) -> Any: ...
# END: tensor_annotations annotations for round(...)

def saturate_cast(value, dtype, name=...) -> Any: ...
def scalar_mul(scalar, x, name=...) -> Any: ...
def scan(fn, elems, initializer=..., parallel_iterations=..., back_prop=..., swap_memory=..., infer_shape=..., reverse=..., name=...) -> Any: ...
def scatter_nd(indices, updates, shape, name=...) -> Any: ...
def searchsorted(sorted_sequence, values, side=..., out_type=..., name=...) -> Any: ...
def sequence_mask(lengths, maxlen=..., dtype=..., name=...) -> Any: ...
def shape(input, out_type=..., name=...) -> Any: ...
def shape_n(input, out_type=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for sigmoid(...)
@overload
def sigmoid(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def sigmoid(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def sigmoid(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def sigmoid(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def sigmoid(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def sigmoid(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def sigmoid(x, name=...) -> Any: ...
# END: tensor_annotations annotations for sigmoid(...)


# BEGIN: tensor_annotations annotations for sign(...)
@overload
def sign(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def sign(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def sign(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def sign(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def sign(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def sign(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def sign(x, name=...) -> Any: ...
# END: tensor_annotations annotations for sign(...)


# BEGIN: tensor_annotations annotations for sin(...)
@overload
def sin(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def sin(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def sin(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def sin(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def sin(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def sin(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def sin(x, name=...) -> Any: ...
# END: tensor_annotations annotations for sin(...)


# BEGIN: tensor_annotations annotations for sinh(...)
@overload
def sinh(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def sinh(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def sinh(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def sinh(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def sinh(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def sinh(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def sinh(x, name=...) -> Any: ...
# END: tensor_annotations annotations for sinh(...)

def size(input, out_type=..., name=...) -> Any: ...
def slice(input_, begin, size, name=...) -> Any: ...
def sort(values, axis=..., direction=..., name=...) -> Any: ...
def space_to_batch(input, block_shape, paddings, name=...) -> Any: ...
def space_to_batch_nd(input, block_shape, paddings, name=...) -> Any: ...
def split(value, num_or_size_splits, axis=..., num=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for sqrt(...)
@overload
def sqrt(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def sqrt(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def sqrt(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def sqrt(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def sqrt(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def sqrt(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def sqrt(x, name=...) -> Any: ...
# END: tensor_annotations annotations for sqrt(...)


# BEGIN: tensor_annotations annotations for square(...)
@overload
def square(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def square(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def square(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def square(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def square(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def square(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def square(x, name=...) -> Any: ...
# END: tensor_annotations annotations for square(...)

def squeeze(input, axis=..., name=...) -> Any: ...
def stack(values, axis=..., name=...) -> Any: ...
def stop_gradient(input, name=...) -> Any: ...
def strided_slice(input_, begin, end, strides=..., begin_mask=..., end_mask=..., ellipsis_mask=..., new_axis_mask=..., shrink_axis_mask=..., var=..., name=...) -> Any: ...
def subtract(x, y, name=...) -> Any: ...
def switch_case(branch_index, branch_fns, default=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for tan(...)
@overload
def tan(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def tan(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def tan(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def tan(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def tan(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def tan(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def tan(x, name=...) -> Any: ...
# END: tensor_annotations annotations for tan(...)


# BEGIN: tensor_annotations annotations for tanh(...)
@overload
def tanh(x: Tensor0[DT], name=...) -> Tensor0[DT]: ...

@overload
def tanh(x: Tensor1[DT, A1], name=...) -> Tensor1[DT, A1]: ...

@overload
def tanh(x: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def tanh(x: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def tanh(x: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def tanh(x: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def tanh(x, name=...) -> Any: ...
# END: tensor_annotations annotations for tanh(...)

def tensor_scatter_nd_add(tensor, indices, updates, name=...) -> Any: ...
def tensor_scatter_nd_max(tensor, indices, updates, name=...) -> Any: ...
def tensor_scatter_nd_min(tensor, indices, updates, name=...) -> Any: ...
def tensor_scatter_nd_sub(tensor, indices, updates, name=...) -> Any: ...
def tensor_scatter_nd_update(tensor, indices, updates, name=...) -> Any: ...
def tensordot(a, b, axes, name=...) -> Any: ...
def tile(input, multiples, name=...) -> Any: ...
def timestamp(name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for transpose(...)
@overload
def transpose(a: Tensor2[DT, A1, A2], name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def transpose(a: Tensor2[DT, A1, A2], perm: Tuple[L0, L1],
              name=...) -> Tensor2[DT, A1, A2]: ...

@overload
def transpose(a: Tensor2[DT, A1, A2], perm: Tuple[L1, L0],
              name=...) -> Tensor2[DT, A2, A1]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], perm: Tuple[L0, L1, L2],
              name=...) -> Tensor3[DT, A1, A2, A3]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], perm: Tuple[L0, L2, L1],
              name=...) -> Tensor3[DT, A1, A3, A2]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], perm: Tuple[L1, L0, L2],
              name=...) -> Tensor3[DT, A2, A1, A3]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], perm: Tuple[L1, L2, L0],
              name=...) -> Tensor3[DT, A2, A3, A1]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], perm: Tuple[L2, L0, L1],
              name=...) -> Tensor3[DT, A3, A1, A2]: ...

@overload
def transpose(a: Tensor3[DT, A1, A2, A3], perm: Tuple[L2, L1, L0],
              name=...) -> Tensor3[DT, A3, A2, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L0, L1, L2, L3],
              name=...) -> Tensor4[DT, A1, A2, A3, A4]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L0, L1, L3, L2],
              name=...) -> Tensor4[DT, A1, A2, A4, A3]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L0, L2, L1, L3],
              name=...) -> Tensor4[DT, A1, A3, A2, A4]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L0, L2, L3, L1],
              name=...) -> Tensor4[DT, A1, A3, A4, A2]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L0, L3, L1, L2],
              name=...) -> Tensor4[DT, A1, A4, A2, A3]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L0, L3, L2, L1],
              name=...) -> Tensor4[DT, A1, A4, A3, A2]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L1, L0, L2, L3],
              name=...) -> Tensor4[DT, A2, A1, A3, A4]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L1, L0, L3, L2],
              name=...) -> Tensor4[DT, A2, A1, A4, A3]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L1, L2, L0, L3],
              name=...) -> Tensor4[DT, A2, A3, A1, A4]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L1, L2, L3, L0],
              name=...) -> Tensor4[DT, A2, A3, A4, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L1, L3, L0, L2],
              name=...) -> Tensor4[DT, A2, A4, A1, A3]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L1, L3, L2, L0],
              name=...) -> Tensor4[DT, A2, A4, A3, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L2, L0, L1, L3],
              name=...) -> Tensor4[DT, A3, A1, A2, A4]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L2, L0, L3, L1],
              name=...) -> Tensor4[DT, A3, A1, A4, A2]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L2, L1, L0, L3],
              name=...) -> Tensor4[DT, A3, A2, A1, A4]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L2, L1, L3, L0],
              name=...) -> Tensor4[DT, A3, A2, A4, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L2, L3, L0, L1],
              name=...) -> Tensor4[DT, A3, A4, A1, A2]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L2, L3, L1, L0],
              name=...) -> Tensor4[DT, A3, A4, A2, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L3, L0, L1, L2],
              name=...) -> Tensor4[DT, A4, A1, A2, A3]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L3, L0, L2, L1],
              name=...) -> Tensor4[DT, A4, A1, A3, A2]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L3, L1, L0, L2],
              name=...) -> Tensor4[DT, A4, A2, A1, A3]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L3, L1, L2, L0],
              name=...) -> Tensor4[DT, A4, A2, A3, A1]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L3, L2, L0, L1],
              name=...) -> Tensor4[DT, A4, A3, A1, A2]: ...

@overload
def transpose(a: Tensor4[DT, A1, A2, A3, A4], perm: Tuple[L3, L2, L1, L0],
              name=...) -> Tensor4[DT, A4, A3, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L1, L2, L3, L4],
              name=...) -> Tensor5[DT, A1, A2, A3, A4, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L1, L2, L4, L3],
              name=...) -> Tensor5[DT, A1, A2, A3, A5, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L1, L3, L2, L4],
              name=...) -> Tensor5[DT, A1, A2, A4, A3, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L1, L3, L4, L2],
              name=...) -> Tensor5[DT, A1, A2, A4, A5, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L1, L4, L2, L3],
              name=...) -> Tensor5[DT, A1, A2, A5, A3, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L1, L4, L3, L2],
              name=...) -> Tensor5[DT, A1, A2, A5, A4, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L2, L1, L3, L4],
              name=...) -> Tensor5[DT, A1, A3, A2, A4, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L2, L1, L4, L3],
              name=...) -> Tensor5[DT, A1, A3, A2, A5, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L2, L3, L1, L4],
              name=...) -> Tensor5[DT, A1, A3, A4, A2, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L2, L3, L4, L1],
              name=...) -> Tensor5[DT, A1, A3, A4, A5, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L2, L4, L1, L3],
              name=...) -> Tensor5[DT, A1, A3, A5, A2, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L2, L4, L3, L1],
              name=...) -> Tensor5[DT, A1, A3, A5, A4, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L3, L1, L2, L4],
              name=...) -> Tensor5[DT, A1, A4, A2, A3, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L3, L1, L4, L2],
              name=...) -> Tensor5[DT, A1, A4, A2, A5, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L3, L2, L1, L4],
              name=...) -> Tensor5[DT, A1, A4, A3, A2, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L3, L2, L4, L1],
              name=...) -> Tensor5[DT, A1, A4, A3, A5, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L3, L4, L1, L2],
              name=...) -> Tensor5[DT, A1, A4, A5, A2, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L3, L4, L2, L1],
              name=...) -> Tensor5[DT, A1, A4, A5, A3, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L4, L1, L2, L3],
              name=...) -> Tensor5[DT, A1, A5, A2, A3, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L4, L1, L3, L2],
              name=...) -> Tensor5[DT, A1, A5, A2, A4, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L4, L2, L1, L3],
              name=...) -> Tensor5[DT, A1, A5, A3, A2, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L4, L2, L3, L1],
              name=...) -> Tensor5[DT, A1, A5, A3, A4, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L4, L3, L1, L2],
              name=...) -> Tensor5[DT, A1, A5, A4, A2, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L0, L4, L3, L2, L1],
              name=...) -> Tensor5[DT, A1, A5, A4, A3, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L0, L2, L3, L4],
              name=...) -> Tensor5[DT, A2, A1, A3, A4, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L0, L2, L4, L3],
              name=...) -> Tensor5[DT, A2, A1, A3, A5, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L0, L3, L2, L4],
              name=...) -> Tensor5[DT, A2, A1, A4, A3, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L0, L3, L4, L2],
              name=...) -> Tensor5[DT, A2, A1, A4, A5, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L0, L4, L2, L3],
              name=...) -> Tensor5[DT, A2, A1, A5, A3, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L0, L4, L3, L2],
              name=...) -> Tensor5[DT, A2, A1, A5, A4, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L2, L0, L3, L4],
              name=...) -> Tensor5[DT, A2, A3, A1, A4, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L2, L0, L4, L3],
              name=...) -> Tensor5[DT, A2, A3, A1, A5, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L2, L3, L0, L4],
              name=...) -> Tensor5[DT, A2, A3, A4, A1, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L2, L3, L4, L0],
              name=...) -> Tensor5[DT, A2, A3, A4, A5, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L2, L4, L0, L3],
              name=...) -> Tensor5[DT, A2, A3, A5, A1, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L2, L4, L3, L0],
              name=...) -> Tensor5[DT, A2, A3, A5, A4, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L3, L0, L2, L4],
              name=...) -> Tensor5[DT, A2, A4, A1, A3, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L3, L0, L4, L2],
              name=...) -> Tensor5[DT, A2, A4, A1, A5, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L3, L2, L0, L4],
              name=...) -> Tensor5[DT, A2, A4, A3, A1, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L3, L2, L4, L0],
              name=...) -> Tensor5[DT, A2, A4, A3, A5, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L3, L4, L0, L2],
              name=...) -> Tensor5[DT, A2, A4, A5, A1, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L3, L4, L2, L0],
              name=...) -> Tensor5[DT, A2, A4, A5, A3, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L4, L0, L2, L3],
              name=...) -> Tensor5[DT, A2, A5, A1, A3, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L4, L0, L3, L2],
              name=...) -> Tensor5[DT, A2, A5, A1, A4, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L4, L2, L0, L3],
              name=...) -> Tensor5[DT, A2, A5, A3, A1, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L4, L2, L3, L0],
              name=...) -> Tensor5[DT, A2, A5, A3, A4, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L4, L3, L0, L2],
              name=...) -> Tensor5[DT, A2, A5, A4, A1, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L1, L4, L3, L2, L0],
              name=...) -> Tensor5[DT, A2, A5, A4, A3, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L0, L1, L3, L4],
              name=...) -> Tensor5[DT, A3, A1, A2, A4, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L0, L1, L4, L3],
              name=...) -> Tensor5[DT, A3, A1, A2, A5, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L0, L3, L1, L4],
              name=...) -> Tensor5[DT, A3, A1, A4, A2, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L0, L3, L4, L1],
              name=...) -> Tensor5[DT, A3, A1, A4, A5, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L0, L4, L1, L3],
              name=...) -> Tensor5[DT, A3, A1, A5, A2, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L0, L4, L3, L1],
              name=...) -> Tensor5[DT, A3, A1, A5, A4, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L1, L0, L3, L4],
              name=...) -> Tensor5[DT, A3, A2, A1, A4, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L1, L0, L4, L3],
              name=...) -> Tensor5[DT, A3, A2, A1, A5, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L1, L3, L0, L4],
              name=...) -> Tensor5[DT, A3, A2, A4, A1, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L1, L3, L4, L0],
              name=...) -> Tensor5[DT, A3, A2, A4, A5, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L1, L4, L0, L3],
              name=...) -> Tensor5[DT, A3, A2, A5, A1, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L1, L4, L3, L0],
              name=...) -> Tensor5[DT, A3, A2, A5, A4, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L3, L0, L1, L4],
              name=...) -> Tensor5[DT, A3, A4, A1, A2, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L3, L0, L4, L1],
              name=...) -> Tensor5[DT, A3, A4, A1, A5, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L3, L1, L0, L4],
              name=...) -> Tensor5[DT, A3, A4, A2, A1, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L3, L1, L4, L0],
              name=...) -> Tensor5[DT, A3, A4, A2, A5, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L3, L4, L0, L1],
              name=...) -> Tensor5[DT, A3, A4, A5, A1, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L3, L4, L1, L0],
              name=...) -> Tensor5[DT, A3, A4, A5, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L4, L0, L1, L3],
              name=...) -> Tensor5[DT, A3, A5, A1, A2, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L4, L0, L3, L1],
              name=...) -> Tensor5[DT, A3, A5, A1, A4, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L4, L1, L0, L3],
              name=...) -> Tensor5[DT, A3, A5, A2, A1, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L4, L1, L3, L0],
              name=...) -> Tensor5[DT, A3, A5, A2, A4, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L4, L3, L0, L1],
              name=...) -> Tensor5[DT, A3, A5, A4, A1, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L2, L4, L3, L1, L0],
              name=...) -> Tensor5[DT, A3, A5, A4, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L0, L1, L2, L4],
              name=...) -> Tensor5[DT, A4, A1, A2, A3, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L0, L1, L4, L2],
              name=...) -> Tensor5[DT, A4, A1, A2, A5, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L0, L2, L1, L4],
              name=...) -> Tensor5[DT, A4, A1, A3, A2, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L0, L2, L4, L1],
              name=...) -> Tensor5[DT, A4, A1, A3, A5, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L0, L4, L1, L2],
              name=...) -> Tensor5[DT, A4, A1, A5, A2, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L0, L4, L2, L1],
              name=...) -> Tensor5[DT, A4, A1, A5, A3, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L1, L0, L2, L4],
              name=...) -> Tensor5[DT, A4, A2, A1, A3, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L1, L0, L4, L2],
              name=...) -> Tensor5[DT, A4, A2, A1, A5, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L1, L2, L0, L4],
              name=...) -> Tensor5[DT, A4, A2, A3, A1, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L1, L2, L4, L0],
              name=...) -> Tensor5[DT, A4, A2, A3, A5, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L1, L4, L0, L2],
              name=...) -> Tensor5[DT, A4, A2, A5, A1, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L1, L4, L2, L0],
              name=...) -> Tensor5[DT, A4, A2, A5, A3, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L2, L0, L1, L4],
              name=...) -> Tensor5[DT, A4, A3, A1, A2, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L2, L0, L4, L1],
              name=...) -> Tensor5[DT, A4, A3, A1, A5, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L2, L1, L0, L4],
              name=...) -> Tensor5[DT, A4, A3, A2, A1, A5]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L2, L1, L4, L0],
              name=...) -> Tensor5[DT, A4, A3, A2, A5, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L2, L4, L0, L1],
              name=...) -> Tensor5[DT, A4, A3, A5, A1, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L2, L4, L1, L0],
              name=...) -> Tensor5[DT, A4, A3, A5, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L4, L0, L1, L2],
              name=...) -> Tensor5[DT, A4, A5, A1, A2, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L4, L0, L2, L1],
              name=...) -> Tensor5[DT, A4, A5, A1, A3, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L4, L1, L0, L2],
              name=...) -> Tensor5[DT, A4, A5, A2, A1, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L4, L1, L2, L0],
              name=...) -> Tensor5[DT, A4, A5, A2, A3, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L4, L2, L0, L1],
              name=...) -> Tensor5[DT, A4, A5, A3, A1, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L3, L4, L2, L1, L0],
              name=...) -> Tensor5[DT, A4, A5, A3, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L0, L1, L2, L3],
              name=...) -> Tensor5[DT, A5, A1, A2, A3, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L0, L1, L3, L2],
              name=...) -> Tensor5[DT, A5, A1, A2, A4, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L0, L2, L1, L3],
              name=...) -> Tensor5[DT, A5, A1, A3, A2, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L0, L2, L3, L1],
              name=...) -> Tensor5[DT, A5, A1, A3, A4, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L0, L3, L1, L2],
              name=...) -> Tensor5[DT, A5, A1, A4, A2, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L0, L3, L2, L1],
              name=...) -> Tensor5[DT, A5, A1, A4, A3, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L1, L0, L2, L3],
              name=...) -> Tensor5[DT, A5, A2, A1, A3, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L1, L0, L3, L2],
              name=...) -> Tensor5[DT, A5, A2, A1, A4, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L1, L2, L0, L3],
              name=...) -> Tensor5[DT, A5, A2, A3, A1, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L1, L2, L3, L0],
              name=...) -> Tensor5[DT, A5, A2, A3, A4, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L1, L3, L0, L2],
              name=...) -> Tensor5[DT, A5, A2, A4, A1, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L1, L3, L2, L0],
              name=...) -> Tensor5[DT, A5, A2, A4, A3, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L2, L0, L1, L3],
              name=...) -> Tensor5[DT, A5, A3, A1, A2, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L2, L0, L3, L1],
              name=...) -> Tensor5[DT, A5, A3, A1, A4, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L2, L1, L0, L3],
              name=...) -> Tensor5[DT, A5, A3, A2, A1, A4]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L2, L1, L3, L0],
              name=...) -> Tensor5[DT, A5, A3, A2, A4, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L2, L3, L0, L1],
              name=...) -> Tensor5[DT, A5, A3, A4, A1, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L2, L3, L1, L0],
              name=...) -> Tensor5[DT, A5, A3, A4, A2, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L3, L0, L1, L2],
              name=...) -> Tensor5[DT, A5, A4, A1, A2, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L3, L0, L2, L1],
              name=...) -> Tensor5[DT, A5, A4, A1, A3, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L3, L1, L0, L2],
              name=...) -> Tensor5[DT, A5, A4, A2, A1, A3]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L3, L1, L2, L0],
              name=...) -> Tensor5[DT, A5, A4, A2, A3, A1]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L3, L2, L0, L1],
              name=...) -> Tensor5[DT, A5, A4, A3, A1, A2]: ...

@overload
def transpose(a: Tensor5[DT, A1, A2, A3, A4, A5], perm: Tuple[L4, L3, L2, L1, L0],
              name=...) -> Tensor5[DT, A5, A4, A3, A2, A1]: ...

@overload
def transpose(a, perm=..., conjugate=..., name=...) -> Any: ...
# END: tensor_annotations annotations for transpose(...)

def truediv(x, y, name=...) -> Any: ...
def truncatediv(x, y, name=...) -> Any: ...
def truncatemod(x, y, name=...) -> Any: ...
def tuple(tensors, control_inputs=..., name=...) -> Any: ...
def type_spec_from_value(value) -> Any: ...
def unique(x, out_idx=..., name=...) -> Any: ...
def unique_with_counts(x, out_idx=..., name=...) -> Any: ...
def unravel_index(indices, dims, name=...) -> Any: ...
def unstack(value, num=..., axis=..., name=...) -> Any: ...
def variable_creator_scope(variable_creator) -> Any: ...
def vectorized_map(fn, elems, fallback_to_while_loop=..., warn=...) -> Any: ...
def where(condition, x=..., y=..., name=...) -> Any: ...
def while_loop(cond, body, loop_vars, shape_invariants=..., parallel_iterations=..., back_prop=..., swap_memory=..., maximum_iterations=..., name=...) -> Any: ...

# BEGIN: tensor_annotations annotations for zeros(...)
@overload
def zeros(shape: Shape1, dtype=..., name=...) -> Tensor1[AnyDType, Any]: ...

@overload
def zeros(shape: Shape2, dtype=..., name=...) -> Tensor2[AnyDType, Any, Any]: ...

@overload
def zeros(shape: Shape3, dtype=..., name=...) -> Tensor3[AnyDType, Any, Any, Any]: ...

@overload
def zeros(shape: Shape4, dtype=..., name=...) -> Tensor4[AnyDType, Any, Any, Any, Any]: ...

@overload
def zeros(shape: Shape5, dtype=..., name=...) -> Tensor5[AnyDType, Any, Any, Any, Any, Any]: ...

# Tensor0 is down here because otherwise it'd match shape e.g. Tuple[Any, Any]
# https://github.com/google/pytype/issues/767
# (e.g. `dim = tf.shape_as_list(x); tf.zeros((dim, dim))` would be Tensor0)
@overload
def zeros(shape: Tuple[()], dtype=..., name=...) -> Tensor0[AnyDType]: ...

@overload
def zeros(shape, dtype=..., name=...) -> AnyDType: ...
# END: tensor_annotations annotations for zeros(...)


# BEGIN: tensor_annotations annotations for zeros_like(...)
@overload
def zeros_like(input: Tensor1[AnyDType, A1], dtype=..., name=...) -> Tensor1[AnyDType, A1]: ...

@overload
def zeros_like(input: Tensor2[AnyDType, A1, A2], dtype=..., name=...) -> Tensor2[AnyDType, A1, A2]: ...

@overload
def zeros_like(input: Tensor3[AnyDType, A1, A2, A3], dtype=..., name=...) -> Tensor3[AnyDType, A1, A2, A3]: ...

@overload
def zeros_like(input: Tensor4[AnyDType, A1, A2, A3, A4], dtype=..., name=...) -> Tensor4[AnyDType, A1, A2, A3, A4]: ...

@overload
def zeros_like(input: Tensor5[AnyDType, A1, A2, A3, A4, A5], dtype=..., name=...) -> Tensor5[AnyDType, A1, A2, A3, A4, A5]: ...

@overload
def zeros_like(input, dtype=..., name=...) -> Any: ...
# END: tensor_annotations annotations for zeros_like(...)
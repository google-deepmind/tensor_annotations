"""Decorators for confirming shapes are correct at runtime."""

import copy
import functools
import inspect
import textwrap
import typing
from typing import Any, Dict, Mapping, Tuple

import tensor_annotations.jax as tjax
import tensor_annotations.tensorflow as ttf
import tree


# TODO: Replace this with something easier to maintain.
_TYPES_TO_CHECK = [
    ttf.Tensor0,
    ttf.Tensor1,
    ttf.Tensor2,
    ttf.Tensor3,
    ttf.Tensor4,
    ttf.Tensor5,
    ttf.Tensor6,
    ttf.Tensor7,
    tjax.Array0,
    tjax.Array1,
    tjax.Array2,
    tjax.Array3,
    tjax.Array4,
]


def _is_tensor_type(t):
  if not hasattr(t, '__origin__'):
    # It's not a generic type, so can't be one of the types we care about.
    return False
  # If t is Tensor1[Height], then t.__origin__ == Tensor1.
  if not any(t.__origin__ is generic_type
             for generic_type in _TYPES_TO_CHECK):
    return False
  return True


def verify_runtime_ranks_of_args_and_return(func):
  """Decorator that verifies ranks of arguments and return are correct.

  For example, if an argument `x` is annotated as having type
  `Tensor2[Height, Width]`, we verify that `len(x.shape) == 2`.

  Note that nested argument and return types are not verified.
  For example, if the return type is `Tuple[int, Tensor2[Height, Width]]`,
  we give up and do no checks.

  Args:
    func: The function to decorate.

  Raises:
    TypeError: If rank of return type or any argument is incorrect.

  Returns:
    Decorated function.
  """

  return functools.partial(_verify_runtime_args_and_return_ranks, func)


def _verify_runtime_args_and_return_ranks(func, *args, **kwargs):
  """Main implementation of verify_runtime_args_and_return_ranks."""
  sig: inspect.Signature = inspect.signature(func)

  # ===== Verify args. =====

  bound_arguments: inspect.BoundArguments = sig.bind(*args, **kwargs)
  # Do some args have default values, but the arg was not specified?
  # If so, set the args to their default values.
  bound_arguments.apply_defaults()
  arg_value_by_name: Dict[str, Any] = bound_arguments.arguments
  arg_signature_by_name: Mapping[str, inspect.Parameter] = sig.parameters

  # Note: we iterate over signatures, not argument values, because some
  # arguments may not have signatures.
  for arg_name in arg_signature_by_name:
    arg_signature = arg_signature_by_name[arg_name]
    arg_value = arg_value_by_name[arg_name]

    arg_type = arg_signature.annotation
    if not _is_tensor_type(arg_type):
      continue

    if not hasattr(arg_value, 'shape'):
      message = textwrap.dedent(f"""\
      Function '{func.__name__}': argument '{arg_name}' has type
      annotation '{arg_type}', but actual type is
      '{type(arg_value).__name__}'.
      """)
      message_one_line = message.replace('\n', ' ')
      raise TypeError(message_one_line)

    # If arg_type is Tensor2[Height, Width],
    # then arg_type.__args__ == (Height, Width).
    annotated_arg_rank = len(arg_type.__args__)
    actual_arg_rank = len(arg_value.shape)

    if annotated_arg_rank != actual_arg_rank:
      arg_name = arg_signature.name
      message = textwrap.dedent(f"""\
      Function '{func.__name__}': argument '{arg_name}' has type
      annotation '{arg_type}' with rank {annotated_arg_rank},
      but actual shape is '{arg_value.shape}' with rank {actual_arg_rank}
      """)
      message_one_line = message.replace('\n', ' ')
      raise TypeError(message_one_line)

  # ===== Call function. =====

  func_return_value = func(*args, **kwargs)

  # ===== Verify return. =====

  return_type = sig.return_annotation
  if not _is_tensor_type(return_type):
    return func_return_value

  if not hasattr(func_return_value, 'shape'):
    message = textwrap.dedent(f"""\
    Function '{func.__name__}': return has type annotation '{return_type}'
    but actual return type is '{type(func_return_value).__name__}'
    """)
    message_one_line = message.replace('\n', ' ')
    raise TypeError(message_one_line)

  annotated_rank = len(return_type.__args__)
  actual_rank = len(func_return_value.shape)
  if annotated_rank != actual_rank:
    message = textwrap.dedent(f"""\
    Function '{func.__name__}': return has type annotation '{return_type}'
    with rank {annotated_rank}, but actual shape is
    '{func_return_value.shape}' with rank {actual_rank}
    """)
    message_one_line = message.replace('\n', ' ')
    raise TypeError(message_one_line)

  return func_return_value


def _is_typed_tuple(x):
  return (
      hasattr(x, '__origin__')
      # `Tuple` for e.g. Python 3.6, `tuple` for e.g. Python 3.9.
      and x.__origin__ in (Tuple, tuple)
  )


def _is_typed_dict(x):
  return (
      hasattr(x, '__origin__')
      # `Dict` for e.g. Python 3.6, `dict` for e.g. Python 3.9.
      and x.__origin__ in (Dict, dict)
  )


def _is_typed_namedtuple(x):
  return hasattr(x, '_fields') and (
      hasattr(x, '_field_types')  # Python 3.6
      or hasattr(x, '__annotations__')  # Python 3.9
  )


def _tree_type_to_type_tree(tree_type: Any) -> Any:
  """Converts a tree-like type to a tree of the component types.

  Examples:

    T = Tuple[int, str]
    tree_type_to_type_tree(T) == (int, str)

    T2 = Dict[str, Tuple[float]]
    tree_type_to_type_tree(T2) == {str: (float,)}

    T3 = List[bool]
    tree_type_to_type_tree(T3) == [bool]

    class T3(NamedTuple):
      obses: Tuple[np.ndarray]
      actions: Tuple[np.ndarray]
    tree_type_to_type_tree(T3) == T3(obses=(np.ndarray,), actions=(np.ndarray,))

  If any of the items in the tree is unparameterised, it is not converted:

    T = Tuple[List, str]
    tree_type_to_type_tree(T) == (List, str)

  Args:
    tree_type: The tree-like type to convert.
  Returns:
    A tree of the component types.
  """
  def convert_tuple(x):
    if not _is_typed_tuple(x):
      return x
    # Check for unparameterised Tuple.
    if (
        not hasattr(x, '__args__') or  # Python 3.9
        x.__args__ is None or  # Python 3.6
        not x.__args__  # Python 3.7
    ):
      return x
    # If x is Tuple[int, str, float], x.__args__ will be (int, str, float).
    args = x.__args__
    # Check for Tuple[()].
    if args == ((),):
      return ()
    return args

  def convert_dict(x):
    if not _is_typed_dict(x):
      return x
    # Check for unparameterised Dict.
    if (
        not hasattr(x, '__args__') or  # Python 3.9
        x.__args__ is None or  # Python 3.6
        # Python 3.7
        x.__args__ == (typing.KT, typing.VT)  # pytype: disable=module-attr
    ):
      return x
    # If x is Dict[str, int], then x.__args__ should be (str, int).
    key_type, value_type = x.__args__
    return {key_type: value_type}

  def convert_named_tuple(x):
    if not _is_typed_namedtuple(x):
      return x
    try:
      # Python 3.6/3.7
      args = dict(x._field_types)  # pylint: disable=protected-access
    except AttributeError:
      # Python 3.9
      args = x.__annotations__
    return x(**args)

  type_tree = tree_type
  # Right now, `type_tree` doesn't even look like a tree.
  # So first, we have to try and convert the top-level type to a tree,
  # e.g. Tuple[Tuple[int]] -> (Tuple[int],)
  for f in (convert_tuple, convert_dict, convert_named_tuple):
    type_tree = f(type_tree)

  # Now we just have to keep converting elements of the tree until all
  # elements have been converted.
  prev_type_tree = copy.deepcopy(type_tree)
  while True:
    for f in (convert_tuple, convert_dict, convert_named_tuple):
      type_tree = tree.map_structure(f, type_tree)
    if type_tree == prev_type_tree:
      break
    prev_type_tree = type_tree

  return type_tree

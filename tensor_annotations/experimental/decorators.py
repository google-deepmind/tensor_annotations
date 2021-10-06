"""Decorators for confirming shapes are correct at runtime."""

import copy
import functools
import inspect
import textwrap
import typing
from typing import Any, Dict, Mapping, Tuple, Union

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


def verify_runtime_ranks_of_args_and_return(func=None, *, check_trees=False):
  """Decorator that verifies ranks of arguments and return are correct.

  For example, if an argument `x` is annotated as having type
  `Tensor2[Height, Width]`, we verify that `len(x.shape) == 2`.

  Note that nested argument and return types are not verified.
  For example, if the return type is `Tuple[int, Tensor2[Height, Width]]`,
  we give up and do no checks.

  Args:
    func: The function to decorate.
    check_trees: Whether to recursively check tree-like types. If `True`, we'll
      recurse through tree elements, and check any node that has a `.shape`
      attribute. We support trees composed of dictionaries, tuples, and
      named tuples.

  Raises:
    TypeError: If rank of return type or any argument is incorrect.

  Returns:
    Decorated function.
  """

  if func is not None:
    # Decorator used with no arguments.
    return functools.partial(_verify_runtime_args_and_return_ranks,
                             func, check_trees)
  else:
    # Decorator used with `check_trees` set explicitly.
    def decorator(func):
      return functools.partial(_verify_runtime_args_and_return_ranks,
                               func, check_trees)
    return decorator


def _verify_runtime_args_and_return_ranks(func, _check_trees, *args, **kwargs):  # pylint: disable=invalid-name
  """Main implementation of verify_runtime_args_and_return_ranks.

  Args:
    func: The function to decorate.
    _check_trees: Whether to check tree-like types. (Underscored to prevent
      name collisions with other arguments in `args` and `kwargs`.)
    *args: Positional arguments to `func`.
    **kwargs: Keyword arguments to `func`.

  Returns:
    The return value of `func`.
  """
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
    if _is_tensor_type(arg_type):
      _check_non_tree(func.__name__, arg_name, arg_value, arg_type)
    elif _is_tree_type(arg_type):
      type_tree = _tree_type_to_type_tree(arg_type)
      _check_tree(func.__name__, arg_name, arg_value, type_tree)

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


def _is_tree_type(x):
  return _is_typed_tuple(x) or _is_typed_dict(x) or _is_typed_namedtuple(x)


def _check_non_tree(
    func_name: str,
    arg_name: str,
    arg_value: Any,
    arg_type: Any,
):
  """Checks a non-tree argument.

  Args:
    func_name: The name of the function whose argument we're checking.
    arg_name: The name of the argument we're checking.
    arg_value: The value of the argument.
    arg_type: The annotated type of the argument.

  Raises:
    TypeError: If the type or rank of `value_tree_subtree` is not what it's
      supposed to be, according to the type from `type_tree`.
  """
  if not hasattr(arg_value, 'shape'):
    message = textwrap.dedent(f"""\
    Function '{func_name}': argument '{arg_name}' has type
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
    message = textwrap.dedent(f"""\
    Function '{func_name}': argument '{arg_name}' has type
    annotation '{arg_type}' with rank {annotated_arg_rank},
    but actual shape is '{arg_value.shape}' with rank {actual_arg_rank}
    """)
    message_one_line = message.replace('\n', ' ')
    raise TypeError(message_one_line)


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
  Raises:
    ValueError: If `tree_type` isn't a tree-like type.
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

  tree_of_types = tree_type
  # Right now, `type_tree` doesn't even look like a tree.
  # So first, we have to try and convert the top-level type to a tree,
  # e.g. Tuple[Tuple[int]] -> (Tuple[int],)
  for f in (convert_tuple, convert_dict, convert_named_tuple):
    tree_of_types = f(tree_of_types)
  if tree_of_types == tree_type:
    raise ValueError('tree_type does not appear to be a tree-like type')

  # Now we just have to keep converting elements of the tree until all
  # elements have been converted.
  prev_type_tree = copy.deepcopy(tree_of_types)
  while True:
    for f in (convert_tuple, convert_dict, convert_named_tuple):
      tree_of_types = tree.map_structure(f, tree_of_types)
    if tree_of_types == prev_type_tree:
      break
    prev_type_tree = tree_of_types

  return tree_of_types


def _check_tree(
    func_name: str,
    arg_name: str,
    value_tree,
    type_tree,
):
  """Checks ranks in a tree-like argument.

  Arguments:
    func_name: The name of the function whose argument we're checking.
    arg_name: The name of the argument we're checking.
    value_tree: The value of the argument.
    type_tree: The types of `value_tree`.
  """
  tree.traverse_with_path(
      functools.partial(_check_tree_traverse, func_name, arg_name, type_tree),
      value_tree,
  )


def _check_tree_traverse(
    func_name: str,
    arg_name: str,
    type_tree,
    path: Tuple[Union[int, str]],
    value_tree_subtree,
):
  """Visits a node of `value_tree`, checking the type from `type_tree`.

  Called from `_check_tree`.

  Args:
    func_name: The name of the function whose argument we're checking.
    arg_name: The name of the argument we're checking.
    type_tree: The types of `value_tree`.
    path: A sequence of the branch keys we had to take to get to where we are.
      For example, if `value_tree` is {'a': (10, 11)}, and if we're at the
      10, then `path` would be ('a', 0).
    value_tree_subtree: The subtree of `value_tree` rooted at the current
      position.

  Raises:
    ValueError: If something goes wrong while trying to find the expected type
      of the current node in `type_tree`.
    TypeError: If the type or rank of `value_tree_subtree` is not what it's
      supposed to be, according to the type from `type_tree`.
  """

  # ===== Step 1: Find the type of this node in `type_tree`. =====

  type_tree_node = type_tree
  path_str = ''
  for path_element in path:
    if isinstance(type_tree_node, Dict):
      if len(type_tree_node) != 1:
        raise ValueError('Expected type tree type_tree_node to be of form '
                         '{key_type: value_type}, but is actually '
                         + str(type_tree_node))
      # If `value_tree` is `{'a': 0}`, then `type_tree` will be `{str: int}`.
      path_str += f"['{path_element}']"
      type_tree_node = next(iter(type_tree_node.values()))
    elif _is_typed_namedtuple(type_tree_node):
      path_str += f'.{path_element}'
      type_tree_node = getattr(type_tree_node, path_element)
    elif isinstance(type_tree_node, Tuple):
      path_str += f'[{path_element}]'
      type_tree_node = type_tree_node[path_element]
    else:
      raise ValueError('Not sure how to descend into type tree node '
                       f"'{type_tree_node}'")
  type_tree_subtree = type_tree_node

  # ===== Step 2: Check the rank. =====

  if not _is_tensor_type(type_tree_subtree):
    return
  value = value_tree_subtree
  expected_type = type_tree_subtree

  if not hasattr(value, 'shape'):
    message = textwrap.dedent(f"""\
    Function '{func_name}': argument '{arg_name}{path_str}' has type
    annotation '{expected_type}', but actual type is
    '{type(value).__name__}'.
    """).strip()
    message_one_line = message.replace('\n', ' ')
    raise TypeError(message_one_line)

  # If arg_type is Tensor2[Height, Width],
  # then arg_type.__args__ == (Height, Width).
  expected_rank = len(expected_type.__args__)
  actual_rank = len(value.shape)

  if expected_rank != actual_rank:
    message = textwrap.dedent(f"""\
    Function '{func_name}': argument '{arg_name}{path_str}' has type
    annotation '{expected_type}' with rank {expected_rank},
    but actual shape is '{value.shape}' with rank {actual_rank}
    """).strip()
    message_one_line = message.replace('\n', ' ')
    raise TypeError(message_one_line)

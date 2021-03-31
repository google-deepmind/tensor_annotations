"""Decorators for confirming shapes are correct at runtime."""

import functools
import inspect
import textwrap
from typing import Any, Dict, Mapping

import tensor_annotations.jax as tjax
import tensor_annotations.tensorflow as ttf


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
  # If arg_type is Tensor1[Height], then t.__origin__ == Tensor1.
  if not any(t.__origin__ is tensor_type
             for tensor_type in _TYPES_TO_CHECK):
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
    ValueError: If rank of return type or any argument is incorrect.

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
      raise ValueError(message_one_line)

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
      raise ValueError(message_one_line)

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
    raise ValueError(message_one_line)

  annotated_rank = len(return_type.__args__)
  actual_rank = len(func_return_value.shape)
  if annotated_rank != actual_rank:
    message = textwrap.dedent(f"""\
    Function '{func.__name__}': return has type annotation '{return_type}'
    with rank {annotated_rank}, but actual shape is
    '{func_return_value.shape}' with rank {actual_rank}
    """)
    message_one_line = message.replace('\n', ' ')
    raise ValueError(message_one_line)

  return func_return_value

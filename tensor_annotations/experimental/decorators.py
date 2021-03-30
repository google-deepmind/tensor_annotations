"""Decorators for confirming shapes are correct at runtime."""

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


def verify_runtime_args_and_return_ranks(func):
  """Decorator that verifies ranks of arguments and return are correct.

  For example, if an argument `x` is annotated as having type
  `Tensor2[Height, Width]`, we verify that `len(x.shape) == 2`.

  Args:
    func: The function to decorate.

  Raises:
    ValueError: If rank of return type or any argument is incorrect.

  Returns:
    Decorated function.
  """

  def wrapper(*args, **kwargs):
    # ===== Verify args. =====

    sig: inspect.Signature = inspect.signature(func)
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
      if not hasattr(arg_type, '__origin__'):
        # It's not a generic type, so can't be one of the types we care about.
        continue
      # If arg_type is Tensor1[Height], then arg_type.__origin__ == Tensor1.
      if not any(arg_type.__origin__ is t for t in _TYPES_TO_CHECK):
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

    func_returns = func(*args, **kwargs)

    # TODO: Verify return.

    return func_returns

  return wrapper

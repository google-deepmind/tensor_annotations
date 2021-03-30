"""Decorators for confirming shapes are correct at runtime."""

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

    # TODO: Verify args.

    # ===== Call function. =====

    func_returns = func(*args, **kwargs)

    # TODO: Verify return.

    return func_returns

  return wrapper

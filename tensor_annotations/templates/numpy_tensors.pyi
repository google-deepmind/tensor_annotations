# LINT.IfChange
"""Type stubs for custom NumPy tensor classes.

NOTE: This file is generated from templates/numpy_tensors.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_tensor_template.py \
     --template templates/numpy_tensors.pyi \
     --out numpy.pyi
"""

from typing import TypeVar, Generic
from tensor_annotations.axes import Axis


A1 = TypeVar('A1', bound=Axis)
A2 = TypeVar('A2', bound=Axis)
A3 = TypeVar('A3', bound=Axis)
A4 = TypeVar('A4', bound=Axis)
A5 = TypeVar('A5', bound=Axis)
A6 = TypeVar('A6', bound=Axis)
A7 = TypeVar('A7', bound=Axis)
A8 = TypeVar('A8', bound=Axis)

# We need to define DTypes ourselves rather than use e.g. np.uint8 because
# pytype sees NumPy's own DTypes as `Any``.
class DType: pass

DT = TypeVar('DT', bound=DType)


class Array1(Generic[DT, A1]):
  pass


class Array2(Generic[DT, A1, A2]):
  pass


class Array3(Generic[DT, A1, A2, A3]):
  pass


class Array4(Generic[DT, A1, A2, A3, A4]):
  pass


class Array5(Generic[DT, A1, A2, A3, A4, A5]):
  pass


class Array6(Generic[DT, A1, A2, A3, A4, A5, A6]):
  pass


class Array7(Generic[DT, A1, A2, A3, A4, A5, A6, A7]):
  pass


class Array8(Generic[DT, A1, A2, A3, A4, A5, A6, A7, A8]):
  pass

# LINT.ThenChange(../numpy.pyi)

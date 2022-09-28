# Change Log

## [2.0.0] - 2022-09-27

### Changed

Types are now generic in dtypes as well as shapes. For example, instead of `Array2[Batch, Time]`, one can now write `Array2[uint8, Batch, Time]`.

Note that this is a breaking change. If you want to retain the old behaviour, you can import `ArrayNAnyDType` instead of `ArrayN` - eg:

```python
from tensor_annotations.jax import Array2AnyDType as Array2
```

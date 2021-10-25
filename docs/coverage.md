# Coverage

This page documents which functions we provide shape-aware stubs for.

## TensorFlow

Note that for TensorFlow, although we don't provide *shape-aware* stubs for
functions marked ❌, we do provide normal stubs, so autocomplete etc should still
work.

### `tensorflow`

a-g                                  | g-r                                           | r-z
------------------------------------ | --------------------------------------------- | ---
✅&nbsp;&nbsp;abs                     | ❌&nbsp;&nbsp;gather_nd                        | ✅&nbsp;&nbsp;round
✅&nbsp;&nbsp;acos                    | ❌&nbsp;&nbsp;get_static_value                 | ❌&nbsp;&nbsp;saturate_cast
✅&nbsp;&nbsp;acosh                   | ❌&nbsp;&nbsp;grad_pass_through                | ❌&nbsp;&nbsp;scalar_mul
❌&nbsp;&nbsp;add                     | ❌&nbsp;&nbsp;gradients                        | ❌&nbsp;&nbsp;scan
❌&nbsp;&nbsp;add_n                   | ❌&nbsp;&nbsp;greater                          | ❌&nbsp;&nbsp;scatter_nd
❌&nbsp;&nbsp;argmax                  | ❌&nbsp;&nbsp;greater_equal                    | ❌&nbsp;&nbsp;searchsorted
❌&nbsp;&nbsp;argmin                  | ❌&nbsp;&nbsp;group                            | ❌&nbsp;&nbsp;sequence_mask
❌&nbsp;&nbsp;argsort                 | ❌&nbsp;&nbsp;hessians                         | ❌&nbsp;&nbsp;shape
❌&nbsp;&nbsp;as_dtype                | ❌&nbsp;&nbsp;histogram_fixed_width            | ❌&nbsp;&nbsp;shape_n
❌&nbsp;&nbsp;as_string               | ❌&nbsp;&nbsp;histogram_fixed_width_bins       | ✅&nbsp;&nbsp;sigmoid
✅&nbsp;&nbsp;asin                    | ❌&nbsp;&nbsp;identity                         | ✅&nbsp;&nbsp;sign
✅&nbsp;&nbsp;asinh                   | ❌&nbsp;&nbsp;identity_n                       | ✅&nbsp;&nbsp;sin
✅&nbsp;&nbsp;atan                    | ❌&nbsp;&nbsp;less                             | ✅&nbsp;&nbsp;sinh
❌&nbsp;&nbsp;atan2                   | ❌&nbsp;&nbsp;less_equal                       | ❌&nbsp;&nbsp;size
✅&nbsp;&nbsp;atanh                   | ❌&nbsp;&nbsp;linspace                         | ❌&nbsp;&nbsp;slice
❌&nbsp;&nbsp;batch_to_space          | ❌&nbsp;&nbsp;logical_and                      | ❌&nbsp;&nbsp;sort
❌&nbsp;&nbsp;bitcast                 | ✅&nbsp;&nbsp;logical_not                      | ❌&nbsp;&nbsp;space_to_batch
❌&nbsp;&nbsp;boolean_mask            | ❌&nbsp;&nbsp;logical_or                       | ❌&nbsp;&nbsp;space_to_batch_nd
❌&nbsp;&nbsp;broadcast_dynamic_shape | ❌&nbsp;&nbsp;make_ndarray                     | ❌&nbsp;&nbsp;split
❌&nbsp;&nbsp;broadcast_static_shape  | ❌&nbsp;&nbsp;map_fn                           | ✅&nbsp;&nbsp;sqrt
❌&nbsp;&nbsp;broadcast_to            | ✅&nbsp;&nbsp;matmul                           | ✅&nbsp;&nbsp;square
❌&nbsp;&nbsp;case                    | ❌&nbsp;&nbsp;matrix_square_root               | ❌&nbsp;&nbsp;squeeze
❌&nbsp;&nbsp;cast                    | ❌&nbsp;&nbsp;maximum                          | ❌&nbsp;&nbsp;stack
❌&nbsp;&nbsp;clip_by_global_norm     | ❌&nbsp;&nbsp;meshgrid                         | ❌&nbsp;&nbsp;stop_gradient
❌&nbsp;&nbsp;clip_by_norm            | ❌&nbsp;&nbsp;minimum                          | ❌&nbsp;&nbsp;strided_slice
❌&nbsp;&nbsp;clip_by_value           | ❌&nbsp;&nbsp;multiply                         | ❌&nbsp;&nbsp;subtract
❌&nbsp;&nbsp;complex                 | ✅&nbsp;&nbsp;negative                         | ❌&nbsp;&nbsp;switch_case
❌&nbsp;&nbsp;concat                  | ❌&nbsp;&nbsp;norm                             | ✅&nbsp;&nbsp;tan
❌&nbsp;&nbsp;cond                    | ❌&nbsp;&nbsp;not_equal                        | ✅&nbsp;&nbsp;tanh
❌&nbsp;&nbsp;constant                | ❌&nbsp;&nbsp;one_hot                          | ❌&nbsp;&nbsp;tensor_scatter_nd_add
❌&nbsp;&nbsp;convert_to_tensor       | ✅&nbsp;&nbsp;ones                             | ❌&nbsp;&nbsp;tensor_scatter_nd_max
✅&nbsp;&nbsp;cos                     | ✅&nbsp;&nbsp;ones_like                        | ❌&nbsp;&nbsp;tensor_scatter_nd_min
✅&nbsp;&nbsp;cosh                    | ❌&nbsp;&nbsp;pad                              | ❌&nbsp;&nbsp;tensor_scatter_nd_sub
❌&nbsp;&nbsp;cumsum                  | ❌&nbsp;&nbsp;parallel_stack                   | ❌&nbsp;&nbsp;tensor_scatter_nd_update
❌&nbsp;&nbsp;divide                  | ❌&nbsp;&nbsp;pow                              | ❌&nbsp;&nbsp;tensordot
❌&nbsp;&nbsp;dynamic_partition       | ❌&nbsp;&nbsp;range                            | ❌&nbsp;&nbsp;tile
❌&nbsp;&nbsp;dynamic_stitch          | ❌&nbsp;&nbsp;rank                             | ❌&nbsp;&nbsp;timestamp
❌&nbsp;&nbsp;edit_distance           | ❌&nbsp;&nbsp;realdiv                          | ✅&nbsp;&nbsp;transpose
❌&nbsp;&nbsp;eig                     | ✅&nbsp;&nbsp;reduce_all                       | ❌&nbsp;&nbsp;truediv
❌&nbsp;&nbsp;eigvals                 | ✅&nbsp;&nbsp;reduce_any                       | ❌&nbsp;&nbsp;truncatediv
❌&nbsp;&nbsp;einsum                  | ✅&nbsp;&nbsp;reduce_logsumexp                 | ❌&nbsp;&nbsp;truncatemod
❌&nbsp;&nbsp;equal                   | ✅&nbsp;&nbsp;reduce_max                       | ❌&nbsp;&nbsp;tuple
✅&nbsp;&nbsp;exp                     | ✅&nbsp;&nbsp;reduce_mean                      | ❌&nbsp;&nbsp;unique
❌&nbsp;&nbsp;expand_dims             | ✅&nbsp;&nbsp;reduce_min                       | ❌&nbsp;&nbsp;unique_with_counts
❌&nbsp;&nbsp;extract_volume_patches  | ✅&nbsp;&nbsp;reduce_prod                      | ❌&nbsp;&nbsp;unravel_index
❌&nbsp;&nbsp;eye                     | ✅&nbsp;&nbsp;reduce_sum                       | ❌&nbsp;&nbsp;unstack
❌&nbsp;&nbsp;fill                    | ❌&nbsp;&nbsp;repeat                           | ❌&nbsp;&nbsp;vectorized_map
❌&nbsp;&nbsp;fingerprint             | ❌&nbsp;&nbsp;required_space_to_batch_paddings | ❌&nbsp;&nbsp;where
✅&nbsp;&nbsp;floor                   | ❌&nbsp;&nbsp;reshape                          | ❌&nbsp;&nbsp;while_loop
❌&nbsp;&nbsp;foldl                   | ❌&nbsp;&nbsp;reverse                          | ✅&nbsp;&nbsp;zeros
❌&nbsp;&nbsp;foldr                   | ❌&nbsp;&nbsp;reverse_sequence                 | ✅&nbsp;&nbsp;zeros_like
❌&nbsp;&nbsp;gather                  | ❌&nbsp;&nbsp;roll                             |

### `tensorflow.math`

a-g                               | g-r                                 | r-z
--------------------------------- | ----------------------------------- | ---
✅&nbsp;&nbsp;abs                  | ❌&nbsp;&nbsp;greater_equal          | ❌&nbsp;&nbsp;reduce_max
❌&nbsp;&nbsp;accumulate_n         | ❌&nbsp;&nbsp;igamma                 | ❌&nbsp;&nbsp;reduce_mean
✅&nbsp;&nbsp;acos                 | ❌&nbsp;&nbsp;igammac                | ❌&nbsp;&nbsp;reduce_min
✅&nbsp;&nbsp;acosh                | ❌&nbsp;&nbsp;imag                   | ❌&nbsp;&nbsp;reduce_prod
❌&nbsp;&nbsp;add                  | ❌&nbsp;&nbsp;in_top_k               | ❌&nbsp;&nbsp;reduce_std
❌&nbsp;&nbsp;add_n                | ❌&nbsp;&nbsp;invert_permutation     | ❌&nbsp;&nbsp;reduce_sum
❌&nbsp;&nbsp;angle                | ✅&nbsp;&nbsp;is_finite              | ❌&nbsp;&nbsp;reduce_variance
❌&nbsp;&nbsp;argmax               | ✅&nbsp;&nbsp;is_inf                 | ✅&nbsp;&nbsp;rint
❌&nbsp;&nbsp;argmin               | ✅&nbsp;&nbsp;is_nan                 | ✅&nbsp;&nbsp;round
✅&nbsp;&nbsp;asin                 | ❌&nbsp;&nbsp;is_non_decreasing      | ✅&nbsp;&nbsp;rsqrt
✅&nbsp;&nbsp;asinh                | ❌&nbsp;&nbsp;is_strictly_increasing | ❌&nbsp;&nbsp;scalar_mul
✅&nbsp;&nbsp;atan                 | ❌&nbsp;&nbsp;l2_normalize           | ❌&nbsp;&nbsp;segment_max
❌&nbsp;&nbsp;atan2                | ✅&nbsp;&nbsp;lbeta                  | ❌&nbsp;&nbsp;segment_mean
✅&nbsp;&nbsp;atanh                | ❌&nbsp;&nbsp;less                   | ❌&nbsp;&nbsp;segment_min
❌&nbsp;&nbsp;bessel_i0            | ❌&nbsp;&nbsp;less_equal             | ❌&nbsp;&nbsp;segment_prod
❌&nbsp;&nbsp;bessel_i0e           | ✅&nbsp;&nbsp;lgamma                 | ❌&nbsp;&nbsp;segment_sum
❌&nbsp;&nbsp;bessel_i1            | ✅&nbsp;&nbsp;log                    | ✅&nbsp;&nbsp;sigmoid
❌&nbsp;&nbsp;bessel_i1e           | ✅&nbsp;&nbsp;log1p                  | ✅&nbsp;&nbsp;sign
❌&nbsp;&nbsp;betainc              | ✅&nbsp;&nbsp;log_sigmoid            | ✅&nbsp;&nbsp;sin
❌&nbsp;&nbsp;bincount             | ❌&nbsp;&nbsp;log_softmax            | ✅&nbsp;&nbsp;sinh
❌&nbsp;&nbsp;ceil                 | ❌&nbsp;&nbsp;logical_and            | ❌&nbsp;&nbsp;sobol_sample
❌&nbsp;&nbsp;confusion_matrix     | ✅&nbsp;&nbsp;logical_not            | ❌&nbsp;&nbsp;softmax
❌&nbsp;&nbsp;conj                 | ❌&nbsp;&nbsp;logical_or             | ✅&nbsp;&nbsp;softplus
✅&nbsp;&nbsp;cos                  | ❌&nbsp;&nbsp;logical_xor            | ✅&nbsp;&nbsp;softsign
✅&nbsp;&nbsp;cosh                 | ❌&nbsp;&nbsp;maximum                | ✅&nbsp;&nbsp;sqrt
❌&nbsp;&nbsp;count_nonzero        | ❌&nbsp;&nbsp;minimum                | ✅&nbsp;&nbsp;square
❌&nbsp;&nbsp;cumprod              | ❌&nbsp;&nbsp;mod                    | ❌&nbsp;&nbsp;squared_difference
❌&nbsp;&nbsp;cumsum               | ❌&nbsp;&nbsp;multiply               | ❌&nbsp;&nbsp;subtract
❌&nbsp;&nbsp;cumulative_logsumexp | ❌&nbsp;&nbsp;multiply_no_nan        | ✅&nbsp;&nbsp;tan
❌&nbsp;&nbsp;digamma              | ✅&nbsp;&nbsp;ndtri                  | ✅&nbsp;&nbsp;tanh
❌&nbsp;&nbsp;divide               | ✅&nbsp;&nbsp;negative               | ❌&nbsp;&nbsp;top_k
❌&nbsp;&nbsp;divide_no_nan        | ❌&nbsp;&nbsp;nextafter              | ❌&nbsp;&nbsp;truediv
❌&nbsp;&nbsp;equal                | ❌&nbsp;&nbsp;not_equal              | ❌&nbsp;&nbsp;unsorted_segment_max
✅&nbsp;&nbsp;erf                  | ❌&nbsp;&nbsp;polygamma              | ❌&nbsp;&nbsp;unsorted_segment_mean
✅&nbsp;&nbsp;erfc                 | ❌&nbsp;&nbsp;polyval                | ❌&nbsp;&nbsp;unsorted_segment_min
❌&nbsp;&nbsp;erfcinv              | ❌&nbsp;&nbsp;pow                    | ❌&nbsp;&nbsp;unsorted_segment_prod
✅&nbsp;&nbsp;erfinv               | ❌&nbsp;&nbsp;real                   | ❌&nbsp;&nbsp;unsorted_segment_sqrt_n
✅&nbsp;&nbsp;exp                  | ✅&nbsp;&nbsp;reciprocal             | ❌&nbsp;&nbsp;unsorted_segment_sum
✅&nbsp;&nbsp;expm1                | ✅&nbsp;&nbsp;reciprocal_no_nan      | ❌&nbsp;&nbsp;xdivy
✅&nbsp;&nbsp;floor                | ❌&nbsp;&nbsp;reduce_all             | ❌&nbsp;&nbsp;xlog1py
❌&nbsp;&nbsp;floordiv             | ❌&nbsp;&nbsp;reduce_any             | ❌&nbsp;&nbsp;xlogy
❌&nbsp;&nbsp;floormod             | ❌&nbsp;&nbsp;reduce_euclidean_norm  | ❌&nbsp;&nbsp;zero_fraction
❌&nbsp;&nbsp;greater              | ❌&nbsp;&nbsp;reduce_logsumexp       | ❌&nbsp;&nbsp;zeta

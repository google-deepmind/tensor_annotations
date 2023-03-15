# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Stubs for np.*

NOTE: This file is generated from templates/numpy.pyi.

To regenerate, run the following from the tensor_annotations directory:
   tools/render_numpy_library_template.py
"""

from typing import Any

# Special-cased because JAX tests expect this to not be Any.
class dtype: pass


ALLOW_THREADS: Any

AxisError: Any

BUFSIZE: Any

CLIP: Any

ComplexWarning: Any

DataSource: Any

ERR_CALL: Any

ERR_DEFAULT: Any

ERR_IGNORE: Any

ERR_LOG: Any

ERR_PRINT: Any

ERR_RAISE: Any

ERR_WARN: Any

FLOATING_POINT_SUPPORT: Any

FPE_DIVIDEBYZERO: Any

FPE_INVALID: Any

FPE_OVERFLOW: Any

FPE_UNDERFLOW: Any

False_: Any

Inf: Any

Infinity: Any

MAXDIMS: Any

MAY_SHARE_BOUNDS: Any

MAY_SHARE_EXACT: Any

ModuleDeprecationWarning: Any

NAN: Any

NINF: Any

NZERO: Any

NaN: Any

PINF: Any

PZERO: Any

RAISE: Any

RankWarning: Any

SHIFT_DIVIDEBYZERO: Any

SHIFT_INVALID: Any

SHIFT_OVERFLOW: Any

SHIFT_UNDERFLOW: Any

ScalarType: Any

Tester: Any

TooHardError: Any

True_: Any

UFUNC_BUFSIZE_DEFAULT: Any

UFUNC_PYVALS_NAME: Any

VisibleDeprecationWarning: Any

WRAP: Any

abs: Any

absolute: Any

add: Any

add_docstring: Any

add_newdoc: Any

add_newdoc_ufunc: Any

all: Any

allclose: Any

alltrue: Any

amax: Any

amin: Any

angle: Any

any: Any

append: Any

apply_along_axis: Any

apply_over_axes: Any

arange: Any

arccos: Any

arccosh: Any

arcsin: Any

arcsinh: Any

arctan: Any

arctan2: Any

arctanh: Any

argmax: Any

argmin: Any

argpartition: Any

argsort: Any

argwhere: Any

around: Any

array: Any

array2string: Any

array_equal: Any

array_equiv: Any

array_repr: Any

array_split: Any

array_str: Any

asanyarray: Any

asarray: Any

asarray_chkfinite: Any

ascontiguousarray: Any

asfarray: Any

asfortranarray: Any

asmatrix: Any

atleast_1d: Any

atleast_2d: Any

atleast_3d: Any

average: Any

bartlett: Any

base_repr: Any

binary_repr: Any

bincount: Any

bitwise_and: Any

bitwise_not: Any

bitwise_or: Any

bitwise_xor: Any

blackman: Any

block: Any

bmat: Any

bool_: Any

broadcast: Any

broadcast_arrays: Any

broadcast_shapes: Any

broadcast_to: Any

busday_count: Any

busday_offset: Any

busdaycalendar: Any

byte: Any

byte_bounds: Any

bytes_: Any

c_: Any

can_cast: Any

cast: Any

cbrt: Any

cdouble: Any

ceil: Any

cfloat: Any

char: Any

character: Any

chararray: Any

choose: Any

clip: Any

clongdouble: Any

clongfloat: Any

column_stack: Any

common_type: Any

compare_chararrays: Any

compat: Any

complex128: Any

complex256: Any

complex64: Any

complex_: Any

complexfloating: Any

compress: Any

concatenate: Any

conj: Any

conjugate: Any

convolve: Any

copy: Any

copysign: Any

copyto: Any

corrcoef: Any

correlate: Any

cos: Any

cosh: Any

count_nonzero: Any

cov: Any

cross: Any

csingle: Any

ctypeslib: Any

cumprod: Any

cumproduct: Any

cumsum: Any

datetime64: Any

datetime_as_string: Any

datetime_data: Any

deg2rad: Any

degrees: Any

delete: Any

deprecate: Any

deprecate_with_doc: Any

diag: Any

diag_indices: Any

diag_indices_from: Any

diagflat: Any

diagonal: Any

diff: Any

digitize: Any

disp: Any

divide: Any

divmod: Any

dot: Any

double: Any

dsplit: Any

dstack: Any

e: Any

ediff1d: Any

einsum: Any

einsum_path: Any

emath: Any

empty: Any

empty_like: Any

equal: Any

errstate: Any

euler_gamma: Any

exp: Any

exp2: Any

expand_dims: Any

expm1: Any

extract: Any

eye: Any

fabs: Any

fastCopyAndTranspose: Any

fft: Any

fill_diagonal: Any

find_common_type: Any

finfo: Any

fix: Any

flatiter: Any

flatnonzero: Any

flexible: Any

flip: Any

fliplr: Any

flipud: Any

float128: Any

float16: Any

float32: Any

float64: Any

float_: Any

float_power: Any

floating: Any

floor: Any

floor_divide: Any

fmax: Any

fmin: Any

fmod: Any

format_float_positional: Any

format_float_scientific: Any

format_parser: Any

frexp: Any

from_dlpack: Any

frombuffer: Any

fromfile: Any

fromfunction: Any

fromiter: Any

frompyfunc: Any

fromregex: Any

fromstring: Any

full: Any

full_like: Any

gcd: Any

generic: Any

genfromtxt: Any

geomspace: Any

get_array_wrap: Any

get_include: Any

get_printoptions: Any

getbufsize: Any

geterr: Any

geterrcall: Any

geterrobj: Any

gradient: Any

greater: Any

greater_equal: Any

half: Any

hamming: Any

hanning: Any

heaviside: Any

histogram: Any

histogram2d: Any

histogram_bin_edges: Any

histogramdd: Any

hsplit: Any

hstack: Any

hypot: Any

i0: Any

identity: Any

iinfo: Any

imag: Any

in1d: Any

index_exp: Any

indices: Any

inexact: Any

inf: Any

info: Any

infty: Any

inner: Any

insert: Any

int16: Any

int32: Any

int64: Any

int8: Any

int_: Any

intc: Any

integer: Any

interp: Any

intersect1d: Any

intp: Any

invert: Any

is_busday: Any

isclose: Any

iscomplex: Any

iscomplexobj: Any

isfinite: Any

isfortran: Any

isin: Any

isinf: Any

isnan: Any

isnat: Any

isneginf: Any

isposinf: Any

isreal: Any

isrealobj: Any

isscalar: Any

issctype: Any

issubclass_: Any

issubdtype: Any

issubsctype: Any

iterable: Any

ix_: Any

kaiser: Any

kernel_version: Any

kron: Any

lcm: Any

ldexp: Any

left_shift: Any

less: Any

less_equal: Any

lexsort: Any

lib: Any

linalg: Any

linspace: Any

little_endian: Any

load: Any

loadtxt: Any

log: Any

log10: Any

log1p: Any

log2: Any

logaddexp: Any

logaddexp2: Any

logical_and: Any

logical_not: Any

logical_or: Any

logical_xor: Any

logspace: Any

longcomplex: Any

longdouble: Any

longfloat: Any

longlong: Any

lookfor: Any

ma: Any

mask_indices: Any

mat: Any

math: Any

matmul: Any

matrix: Any

max: Any

maximum: Any

maximum_sctype: Any

may_share_memory: Any

mean: Any

median: Any

memmap: Any

meshgrid: Any

mgrid: Any

min: Any

min_scalar_type: Any

minimum: Any

mintypecode: Any

mod: Any

modf: Any

moveaxis: Any

msort: Any

multiply: Any

nan: Any

nan_to_num: Any

nanargmax: Any

nanargmin: Any

nancumprod: Any

nancumsum: Any

nanmax: Any

nanmean: Any

nanmedian: Any

nanmin: Any

nanpercentile: Any

nanprod: Any

nanquantile: Any

nanstd: Any

nansum: Any

nanvar: Any

nbytes: Any

ndarray: Any

ndenumerate: Any

ndim: Any

ndindex: Any

nditer: Any

negative: Any

nested_iters: Any

newaxis: Any

nextafter: Any

nonzero: Any

not_equal: Any

numarray: Any

number: Any

obj2sctype: Any

object_: Any

ogrid: Any

oldnumeric: Any

ones: Any

ones_like: Any

outer: Any

packbits: Any

pad: Any

partition: Any

percentile: Any

pi: Any

piecewise: Any

place: Any

poly: Any

poly1d: Any

polyadd: Any

polyder: Any

polydiv: Any

polyfit: Any

polyint: Any

polymul: Any

polynomial: Any

polysub: Any

polyval: Any

positive: Any

power: Any

printoptions: Any

prod: Any

product: Any

promote_types: Any

ptp: Any

put: Any

put_along_axis: Any

putmask: Any

quantile: Any

r_: Any

rad2deg: Any

radians: Any

random: Any

ravel: Any

ravel_multi_index: Any

real: Any

real_if_close: Any

rec: Any

recarray: Any

recfromcsv: Any

recfromtxt: Any

reciprocal: Any

record: Any

remainder: Any

repeat: Any

require: Any

reshape: Any

resize: Any

result_type: Any

right_shift: Any

rint: Any

roll: Any

rollaxis: Any

roots: Any

rot90: Any

round: Any

round_: Any

row_stack: Any

s_: Any

safe_eval: Any

save: Any

savetxt: Any

savez: Any

savez_compressed: Any

sctype2char: Any

sctypeDict: Any

sctypes: Any

searchsorted: Any

select: Any

set_numeric_ops: Any

set_printoptions: Any

set_string_function: Any

setbufsize: Any

setdiff1d: Any

seterr: Any

seterrcall: Any

seterrobj: Any

setxor1d: Any

shape: Any

shares_memory: Any

short: Any

show_config: Any

show_runtime: Any

sign: Any

signbit: Any

signedinteger: Any

sin: Any

sinc: Any

single: Any

singlecomplex: Any

sinh: Any

size: Any

sometrue: Any

sort: Any

sort_complex: Any

source: Any

spacing: Any

split: Any

sqrt: Any

square: Any

squeeze: Any

stack: Any

std: Any

str_: Any

string_: Any

subtract: Any

sum: Any

swapaxes: Any

take: Any

take_along_axis: Any

tan: Any

tanh: Any

tensordot: Any

test: Any

testing: Any

tile: Any

timedelta64: Any

trace: Any

tracemalloc_domain: Any

transpose: Any

trapz: Any

tri: Any

tril: Any

tril_indices: Any

tril_indices_from: Any

trim_zeros: Any

triu: Any

triu_indices: Any

triu_indices_from: Any

true_divide: Any

trunc: Any

typecodes: Any

typename: Any

ubyte: Any

ufunc: Any

uint: Any

uint16: Any

uint32: Any

uint64: Any

uint8: Any

uintc: Any

uintp: Any

ulonglong: Any

unicode_: Any

union1d: Any

unique: Any

unpackbits: Any

unravel_index: Any

unsignedinteger: Any

unwrap: Any

use_hugepage: Any

ushort: Any

vander: Any

var: Any

vdot: Any

vectorize: Any

version: Any

void: Any

vsplit: Any

vstack: Any

where: Any

who: Any

zeros: Any

zeros_like: Any


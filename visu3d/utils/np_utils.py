# Copyright 2022 The visu3d Authors.
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

"""Numpy utils.

And utils intended to work on both `xnp.ndarray` and `v3d.DataclassArray`.
"""

from __future__ import annotations

from typing import Any, Union, Optional

import einops
from etils import enp
from etils.array_types import Array, ArrayLike, FloatArray  # pylint: disable=g-multiple-import
from visu3d import array_dataclass
from visu3d.typing import Axes, DcOrArrayT, DTypeArg, Shape  # pylint: disable=g-multiple-import
from visu3d.utils.lazy_imports import scipy

# Maybe some of those could live in `enp` ?


def size_of(shape: Shape) -> int:
  """Returns the size associated with the shape."""
  # TODO(b/198633198): Warning: In TF `bool(shape) == True` for `shape==()`
  if not len(shape):  # pylint: disable=g-explicit-length-test
    size = 1  # Special case because `np.prod([]) == 1.0`
  else:
    size = enp.lazy.np.prod(shape)
  return size


def get_xnp(x: Any) -> enp.NpModule:
  """Returns the np module associated with the given array or DataclassArray."""
  if isinstance(x, array_dataclass.DataclassArray):
    xnp = x.xnp
  elif enp.lazy.is_array(x):
    xnp = enp.lazy.get_xnp(x)
  else:
    raise TypeError(
        f'Unexpected array type: {type(x)}. Could not infer numpy module.')
  return xnp


def is_array(
    x: Any,
    *,
    xnp: Optional[enp.NpModule] = None,
) -> bool:
  """Returns whether `x` is an array or DataclassArray.

  Args:
    x: array to check
    xnp: If given, return False if the array is from a different numpy module.

  Returns:
    True if `x` is `xnp.ndarray` or `v3d.DataclassArray`
  """
  try:
    infered_xnp = get_xnp(x)
  except TypeError:
    return False
  else:
    if xnp is None:
      return True
    else:
      return infered_xnp is xnp


def asarray(
    x: Union[DcOrArrayT, ArrayLike[Array['...']]],
    *,
    xnp: enp.NpModule = None,
    dtype: Optional[DTypeArg] = None,
    optional: bool = False,
) -> DcOrArrayT:
  """Convert `list` to arrays.

  * Validate that x is either `np` or `xnp` (e.g. `np->jnp`, `np->tf` works,
    but not `jnp->tf`, `jnp->np`,...)
  * Dataclass arrays are forwarded.

  Args:
    x: array to check
    xnp: If given, raise an error if the array is from a different numpy module.
      strict
    dtype: If given, cast the array to the dtype
    optional: If True, `x` can be None

  Returns:
    True if `x` is `xnp.ndarray` or `v3d.DataclassArray`
  """
  if x is None:
    if optional:
      return x
    else:
      raise ValueError('Expected array, got `None`')
  if isinstance(x, (int, float, list, tuple)):
    x = xnp.asarray(x, dtype=dtype)

  detected_xnp = get_xnp(x)

  # Only `np` -> `xnp` conversion is accepted
  if detected_xnp is not enp.lazy.np and detected_xnp is not xnp:
    raise TypeError(f'Expected {xnp.__name__} got {detected_xnp.__name__}')

  if isinstance(x, array_dataclass.DataclassArray):
    return x.as_xnp(xnp)
  else:
    return xnp.asarray(x, dtype=dtype)


def to_absolute_axis(axis: Axes, *, ndim: int) -> Axes:
  """Normalize the axis to absolute value.

  Example for self.shape == (x0, x1, x2, x3):

  ```
  to_absolute_axis(None) == (0, 1, 2, 3)
  to_absolute_axis(0) == 0
  to_absolute_axis(-1) == 3
  to_absolute_axis(-2) == 2
  to_absolute_axis((-1, -2)) == (3, 2)
  ```

  Args:
    axis: Axis to normalize
    ndim: Number of dimensions

  Returns:
    The new axis
  """
  if axis is None:
    return tuple(range(ndim))
  elif isinstance(axis, int):
    if axis >= ndim or axis < -ndim:
      raise enp.lazy.np.AxisError(
          axis=axis,
          ndim=ndim,
          # msg_prefix=
          # f'For {self.__class__.__qualname__} with shape={self.shape}',
      )
    elif axis < 0:
      return ndim + axis
    else:
      return axis
  elif isinstance(axis, tuple):
    if not all(isinstance(dim, int) for dim in axis):
      raise ValueError(f'Invalid axis={axis}')
    return tuple(to_absolute_axis(dim, ndim=ndim) for dim in axis)
  else:
    raise TypeError(f'Unexpected axis type: {type(axis)} {axis}')


def to_absolute_einops(shape_pattern: str, *, nlastdim: int) -> str:
  """Convert the einops to absolute."""
  # Nested dataclass might already have shape set.
  offset = 0
  while _einops_dim_name(offset) in shape_pattern:
    offset += 1
  last_dims = [_einops_dim_name(i + offset) for i in range(nlastdim)]
  last_dims = ' '.join(last_dims)
  before, after = shape_pattern.split('->')
  before = f'{before} {last_dims} '
  after = f'{after} {last_dims}'
  return '->'.join([before, after])


def _einops_dim_name(i: int) -> str:
  return f'arr__{i}'


def normalize(x: FloatArray['*d'], axis: int = -1) -> FloatArray['*d']:
  """Normalize the vector to the unit norm."""
  return x / enp.linalg.norm(x, axis=axis, keepdims=True)


def append_row(
    x: FloatArray['*d daxis'],
    value: float,
    *,
    axis: int,  # Axis is required as `row` imply `axis=0` while we want `=-1`
) -> FloatArray['*d daxis+1']:
  """Like `np.append`, but broadcast the value to `x` shape."""
  xnp = enp.get_np_module(x)
  value = xnp.asarray(value)
  if value.ndim == 0:
    shape = list(x.shape)
    shape[axis] = 1
    value = xnp.broadcast_to(value, shape)
  elif value.ndim == 1:
    # TODO(epot): support actual row: append_row(x, [0, 0, 0, 1]). Might require
    # adding a `broadcast_to` which support arbitrary array.
    assert x.shape[axis] == len(value)
    raise NotImplementedError()
  else:
    raise ValueError(
        f'`append_row` does not support appending rank > 1. Got {value.shape}.')
  return xnp.append(x, value, axis=axis)


def interp_points(
    points: FloatArray['num_points d'],
    *,
    t: Union[int, FloatArray['t']],
    axis: int = -1,
    **splprep_kwargs,
) -> FloatArray['t d']:
  """Spline interpolation between x-d Points.

  Args:
    points: Key points of shape `(num_point, num_dims)`
    t: Either number of steps (in which case steps are linearly interpolated),
      or an array of `[0, 1]` values corresponding to the)
    axis: Axis
    **splprep_kwargs: Kwargs forwarded to `scipy.interpolate.splprep`

  Returns:
    The interpolated points of shape `(t, num_dims)`
  """
  if isinstance(points, array_dataclass.DataclassArray):
    # Could eventually add a protocol to support interpolation between Camera,
    # Ray,...
    raise NotImplementedError('DataclassArray not supported. Only `np.array`.')
  xnp = enp.lazy.get_xnp(points, strict=False)
  # TODO(epot): Currently only np is supported
  if xnp is not enp.lazy.np:
    raise NotImplementedError(f'Only numpy supported. Not {xnp}')
  if axis != -1:
    raise NotImplementedError(
        'interpolation currently only supports axis=-1. Open an issue if you '
        'need this.')
  points = xnp.asarray(points)
  points = einops.rearrange(points, 'num_points d -> d num_points')

  tck, _ = scipy.interpolate.splprep(points, **splprep_kwargs)

  if isinstance(t, int):
    t = xnp.linspace(0, 1, t)
  points_fine = scipy.interpolate.splev(t, tck)
  points_fine = einops.rearrange(points_fine, 'd t -> t d')
  return points_fine

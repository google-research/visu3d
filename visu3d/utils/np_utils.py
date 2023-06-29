# Copyright 2023 The visu3d Authors.
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

from typing import TypeVar, Union

import dataclass_array as dca
import einops
from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import

_T = TypeVar('_T')

# Maybe some of those could live in `enp` ?


def append_row(
    x: FloatArray['*d daxis'],
    value: float,
    *,
    axis: int,  # Axis is required as `row` imply `axis=0` while we want `=-1`
) -> FloatArray['*d daxis+1']:
  """Like `np.append`, but broadcast the value to `x` shape."""
  xnp = enp.get_np_module(x)
  value = xnp.asarray(value)
  if len(value.shape) == 0:  # pylint: disable=g-explicit-length-test
    shape = list(x.shape)
    shape[axis] = 1
    value = xnp.broadcast_to(value, shape)
  elif len(value.shape) == 1:
    # TODO(epot): support actual row: append_row(x, [0, 0, 0, 1]). Might require
    # adding a `broadcast_to` which support arbitrary array.
    assert x.shape[axis] == len(value)
    raise NotImplementedError()
  else:
    raise ValueError(
        f'`append_row` does not support appending rank > 1. Got {value.shape}.'
    )
  return enp.compat.concat([x, value], axis=axis)


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
  import scipy.interpolate  # pylint: disable=g-import-not-at-top

  if isinstance(points, dca.DataclassArray):
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
        'need this.'
    )
  points = xnp.asarray(points)
  points = einops.rearrange(points, 'num_points d -> d num_points')

  tck, _ = scipy.interpolate.splprep(points, **splprep_kwargs)

  if isinstance(t, int):
    t = xnp.linspace(0, 1, t)
  points_fine = scipy.interpolate.splev(t, tck)
  points_fine = einops.rearrange(points_fine, 'd t -> t d')
  return points_fine


def __sub__(self: _T, translation: FloatArray['... 3']) -> _T:  # pylint: disable=invalid-name
  """Add `my_obj - array` support, assuming `my_obj + array` exists."""
  translation = self.xnp.asarray(translation)
  return self + (-translation)

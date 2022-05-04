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

"""Tests for np_utils."""

from __future__ import annotations

from etils import enp
import numpy as np
import pytest
import visu3d as v3d
from visu3d.utils import np_utils


@enp.testing.parametrize_xnp()
def test_get_xnp(xnp: enp.NpModule):
  # Dataclass array support
  r = v3d.Ray(pos=xnp.array([3., 0, 0]), dir=xnp.array([3., 0, 0]))
  assert np_utils.get_xnp(r) is xnp
  # Array support
  assert np_utils.get_xnp(xnp.array([3., 0, 0])) is xnp

  with pytest.raises(TypeError, match='Unexpected array type'):
    np_utils.get_xnp('abc')


def test_to_absolute_axis():
  assert np_utils.to_absolute_axis(None, ndim=4) == (0, 1, 2, 3)
  assert np_utils.to_absolute_axis(0, ndim=4) == 0
  assert np_utils.to_absolute_axis(1, ndim=4) == 1
  assert np_utils.to_absolute_axis(2, ndim=4) == 2
  assert np_utils.to_absolute_axis(3, ndim=4) == 3
  assert np_utils.to_absolute_axis(-1, ndim=4) == 3
  assert np_utils.to_absolute_axis(-2, ndim=4) == 2
  assert np_utils.to_absolute_axis(-3, ndim=4) == 1
  assert np_utils.to_absolute_axis(-4, ndim=4) == 0
  assert np_utils.to_absolute_axis((0,), ndim=4) == (0,)
  assert np_utils.to_absolute_axis((0, 1), ndim=4) == (0, 1)
  assert np_utils.to_absolute_axis((0, 1, -1), ndim=4) == (0, 1, 3)
  assert np_utils.to_absolute_axis((-1, -2), ndim=4) == (3, 2)

  with pytest.raises(np.AxisError):
    assert np_utils.to_absolute_axis(4, ndim=4)

  with pytest.raises(np.AxisError):
    assert np_utils.to_absolute_axis(-5, ndim=4)

  with pytest.raises(np.AxisError):
    assert np_utils.to_absolute_axis((0, 4), ndim=4)


def test_to_absolute_einops():
  assert np_utils.to_absolute_einops(
      'b (h w) -> b h w',
      nlastdim=2,
  ) == 'b (h w)  arr__0 arr__1 -> b h w arr__0 arr__1'
  assert np_utils.to_absolute_einops(
      'b (h w) arr__0 -> b h w arr__0',
      nlastdim=2,
  ) == 'b (h w) arr__0  arr__1 arr__2 -> b h w arr__0 arr__1 arr__2'


@enp.testing.parametrize_xnp()
def test_normalize(xnp: enp.NpModule):
  x = xnp.array([3., 0, 0])
  y = np_utils.normalize(x)
  assert isinstance(y, xnp.ndarray)
  assert y.shape == x.shape
  np.testing.assert_allclose(y, [1., 0., 0.])


@enp.testing.parametrize_xnp()
def test_normalize_batched(xnp: enp.NpModule):
  x = xnp.array([
      [3., 0, 0],
      [0, 4., 0],
      [2., 3., 0],
  ])
  y = np_utils.normalize(x)
  assert isinstance(y, xnp.ndarray)
  assert y.shape == x.shape
  norm = np.sqrt(2**2 + 3**2)
  np.testing.assert_allclose(
      y,
      [
          [1., 0, 0],
          [0, 1., 0],
          [2. / norm, 3. / norm, 0],
      ],
  )


@enp.testing.parametrize_xnp()
def test_append_row(xnp: enp.NpModule):
  x = xnp.ones((2, 4))
  y = np_utils.append_row(x, value=4., axis=-1)
  assert isinstance(y, xnp.ndarray)
  expected = [
      [1, 1, 1, 1, 4],
      [1, 1, 1, 1, 4],
  ]
  np.testing.assert_allclose(y, expected)

  y = np_utils.append_row(x, value=4., axis=0)
  assert isinstance(y, xnp.ndarray)
  expected = [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [4, 4, 4, 4],
  ]
  np.testing.assert_allclose(y, expected)


def test_interpolate():
  p = [
      [0, 0, 0],
      [0, 2, 0],
      [0, 2, 1],
      [0, 0, -5],
  ]
  p_interp = v3d.utils.np_utils.interp_points(p, t=100)
  assert p_interp.shape == (100, 3)

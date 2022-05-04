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

"""Tests for ray."""

from __future__ import annotations

from etils import enp
from etils.array_types import Array
import numpy as np
import pytest
import visu3d as v3d

# Activate the fixture
set_tnp = enp.testing.set_tnp

assert not v3d.Ray._v3d_tree_map_registered


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (2,), (2, 3)])
def test_ray(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
):

  def _broadcast(x: Array['*d'], shape=shape) -> Array['*d 3']:
    return xnp.broadcast_to(xnp.asarray(x), shape + (3,))

  def _ray_broadcast(
      t: Array['3'],
      d: Array['3'],
      shape=shape,
  ) -> Array['*d 3']:
    return v3d.Ray(
        pos=_broadcast(t, shape=shape),
        dir=_broadcast(d, shape=shape),
    )

  def _assert_ray_match(p, t, d, shape=shape):
    assert p.xnp is xnp
    assert p.shape == shape
    v3d.testing.assert_allclose(p, _ray_broadcast(t=t, d=d, shape=shape))

  t = xnp.array([1, 0, 0])
  d = xnp.array([0, 2, 2])
  sqrt8 = np.sqrt(0**2 + 2**2 + 2**2.)

  p = _ray_broadcast(t=t, d=d)

  p_brodcast = v3d.Ray(pos=t, dir=d).map_field(_broadcast)
  v3d.testing.assert_allclose(p, p_brodcast)

  _assert_ray_match(p.mean(), t=t, d=d, shape=())
  _assert_ray_match(p + [3, 2, -1], t=[4, 2, -1], d=d)
  _assert_ray_match(p.scale_dir(3), t=t, d=[0, 6, 6])
  _assert_ray_match(p.look_at([-2, 3, 1]), t=t, d=[-3, 3, 1])
  _assert_ray_match(p.normalize(), t=t, d=[0, 2 / sqrt8, 2 / sqrt8])

  norm = p.norm()
  assert norm.shape == shape
  np.testing.assert_allclose(
      norm,
      xnp.broadcast_to(sqrt8, shape),
  )

  end = p.end
  assert end.shape == shape + (3,)
  np.testing.assert_allclose(end, _broadcast(end))

  _ = p.fig


@enp.testing.parametrize_xnp()
def test_ray_mean(xnp: enp.NpModule):
  ray = v3d.Ray(
      pos=[0, 0, 0],
      dir=[1, 1, 1],
  )
  ray = ray.as_xnp(xnp)
  ray = ray.broadcast_to((1, 2, 3, 4))

  assert ray.mean().shape == ()  # pylint: disable=g-explicit-bool-comparison
  assert ray.mean(axis=None).shape == ()  # pylint: disable=g-explicit-bool-comparison
  assert ray.mean(axis=0).shape == (2, 3, 4)
  assert ray.mean(axis=1).shape == (1, 3, 4)
  assert ray.mean(axis=2).shape == (1, 2, 4)
  assert ray.mean(axis=3).shape == (1, 2, 3)
  assert ray.mean(axis=-1).shape == (1, 2, 3)
  assert ray.mean(axis=-4).shape == (2, 3, 4)
  assert ray.mean(axis=(0, -1)).shape == (2, 3)

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

"""Tests for point."""

from __future__ import annotations

from etils import enp
import pytest
import visu3d as v3d

# Activate the fixture
set_tnp = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('with_color', [False, True])
def test_point(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
    with_color: bool,
):
  init_kwargs = {}
  if with_color:
    init_kwargs['rgb'] = xnp.ones(shape=shape + (3,))

  p = v3d.Point3d(p=xnp.ones(shape=shape + (3,)), **init_kwargs)

  tr = v3d.Transform(R=xnp.eye(3), t=xnp.zeros((3,)))
  p2 = tr @ p
  v3d.testing.assert_array_equal(p, p2)

  v3d.testing.assert_array_equal(p + [0, 0, 0], p)
  v3d.testing.assert_array_equal(p + xnp.array([0, 0, 0]), p)

  p_clipped = p.clip(max=[0.5, 0.5, 5])
  p_clipped_expected = v3d.Point3d(p=xnp.array([0.5, 0.5, 1]), **init_kwargs)
  p_clipped_expected = p_clipped_expected.broadcast_to(shape)
  v3d.testing.assert_array_equal(p_clipped, p_clipped_expected)

  # Display should works
  _ = p.fig


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('with_color', [False, True])
@pytest.mark.parametrize('with_depth', [False, True])
def test_point_2d(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
    with_color: bool,
    with_depth: bool,
):
  init_kwargs = {}
  if with_color:
    init_kwargs['rgb'] = xnp.ones(shape=shape + (3,))
  if with_depth:
    init_kwargs['depth'] = xnp.ones(shape=shape + (1,))

  p = v3d.Point2d(p=xnp.ones(shape=shape + (2,)), **init_kwargs)

  # Point2d can be projected back and forth to camera
  spec = v3d.PinholeCamera.from_focal(resolution=(4, 4), focal_in_px=35)
  spec = spec.as_xnp(xnp)

  p3d = spec.cam_from_px @ p
  assert isinstance(p3d, v3d.Point3d)

  p2d = spec.px_from_cam @ p3d
  assert isinstance(p2d, v3d.Point2d)

  # Round-trip projection from/to pixel should be a no-op
  if with_depth:
    v3d.testing.assert_array_equal(p, p2d)

  # Clip
  p_clipped = p.clip(max=[0.5, 5])
  p_clipped_expected = v3d.Point2d(p=xnp.array([0.5, 1]), **init_kwargs)
  p_clipped_expected = p_clipped_expected.broadcast_to(shape)
  v3d.testing.assert_array_equal(p_clipped, p_clipped_expected)

  # Point2d does not support transform (3d)
  tr = v3d.Transform(R=xnp.eye(3), t=xnp.zeros((3,)))
  with pytest.raises(
      NotImplementedError,
      match='does not implement the `.apply_transform',
  ):
    _ = tr @ p

  # Display should works
  _ = p.fig

# Copyright 2024 The visu3d Authors.
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

"""Tests for camera_spec."""

from __future__ import annotations

import dataclass_array as dca
from etils import enp
import numpy as np
import pytest
import visu3d as v3d

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp

H, W = 640, 480


def make_camera_spec(
    *,
    xnp: enp.NpModule,
    shape: dca.typing.Shape,
) -> v3d.PinholeCamera:
  spec = v3d.PinholeCamera.from_focal(
      resolution=(H, W),
      focal_in_px=35.0,
      xnp=xnp,
  )
  spec = spec.broadcast_to(shape)
  return spec


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize('spec_shape', [(), (3,), (2, 3)])
def test_camera_spec_init(
    xnp: enp.NpModule,
    spec_shape: dca.typing.Shape,
):
  if spec_shape:
    dca.testing.skip_vmap_unavailable(xnp)

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)
  assert spec.resolution == (H, W)
  assert spec.h == H
  assert spec.w == W
  assert spec.wh == (W, H)
  assert spec.hw == (H, W)
  assert spec.shape == spec_shape
  assert spec.K.shape == spec_shape + (3, 3)

  if xnp is None:
    xnp = np
  assert spec.xnp is xnp

  x = _broadcast_to(xnp, [0, 0, 1.0], (1,) * len(spec_shape) + (3,))
  assert enp.compat.is_array_xnp(spec.px_from_cam @ x, xnp)

  x = _broadcast_to(xnp, [0, 0.0], (1,) * len(spec_shape) + (2,))
  assert enp.compat.is_array_xnp(spec.cam_from_px @ x, xnp)

  _ = spec.fig

  spec = spec.replace_fig_config(scale=3.0)
  assert spec.fig_config.scale == 3.0
  _ = spec.fig


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('spec_shape', [(), (3,), (2, 3)])
@pytest.mark.parametrize('point_shape', [(), (2, 3)])
def test_camera_spec_central_point(
    xnp: enp.NpModule,
    spec_shape: dca.typing.Shape,
    point_shape: dca.typing.Shape,
):
  if spec_shape:
    dca.testing.skip_vmap_unavailable(xnp)

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)

  # Projecting the central point (batched)
  central_point_cam = _broadcast_to(
      xnp,
      [0, 0, 1.0],
      spec_shape + point_shape + (3,),
  )
  central_point_px = spec.px_from_cam @ central_point_cam
  assert enp.compat.is_array_xnp(central_point_px, xnp)
  assert central_point_px.shape == spec_shape + point_shape + (2,)
  np.testing.assert_allclose(
      central_point_px,
      np.broadcast_to([W / 2, H / 2], spec_shape + point_shape + (2,)),
  )

  # Round trip conversion
  np.testing.assert_allclose(
      central_point_cam,
      spec.cam_from_px @ central_point_px,
      atol=1e-4,
  )


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('spec_shape', [(), (2, 3)])
def test_camera_px_centers(
    xnp: enp.NpModule,
    spec_shape: dca.typing.Shape,
):
  if spec_shape:
    dca.testing.skip_vmap_unavailable(xnp)

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)

  px_centers = spec.px_centers()
  assert enp.compat.is_array_xnp(px_centers, xnp)
  assert px_centers.shape == spec_shape + (H, W, 2)

  cam_centers = spec.cam_from_px @ px_centers
  assert enp.compat.is_array_xnp(cam_centers, xnp)
  assert cam_centers.shape == spec_shape + (H, W, 3)

  round_trip_px = spec.px_from_cam @ cam_centers
  assert enp.compat.is_array_xnp(round_trip_px, xnp)
  assert round_trip_px.shape == spec_shape + (H, W, 2)

  np.testing.assert_allclose(round_trip_px, px_centers, atol=1e-4)
  # Scaling/normalizing don't change the projection
  np.testing.assert_allclose(
      round_trip_px,
      spec.px_from_cam @ (cam_centers * 3.0),
      atol=1e-4,
  )


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('spec_shape', [(), (2, 3)])
def test_camera_points(
    xnp: enp.NpModule,
    spec_shape: dca.typing.Shape,
):
  if spec_shape:
    dca.testing.skip_vmap_unavailable(
        xnp,
        skip_torch='cannot vmap over non-Tensor arguments',  # TODO(epot): Fix
    )
  if spec_shape and xnp is enp.lazy.tnp:
    pytest.skip('`TF fail when class has None values')

  spec = make_camera_spec(xnp=xnp, shape=spec_shape)

  # Random point cloud in camera coordinates
  rng = np.random.default_rng(0)
  coords3d = rng.random(spec_shape + (5, 3), dtype=np.float32)
  rgb = rng.integers(255, size=spec_shape + (5, 3))
  points3d = v3d.Point3d(p=coords3d, rgb=rgb)
  points3d = points3d.as_xnp(xnp)

  px = spec.px_from_cam @ points3d
  assert isinstance(px, v3d.Point2d)
  assert px.shape == spec_shape + (5,)
  assert px.depth is not None

  # Round-trip should be a no-op (as depth is preserved)
  round_trip_points3d = spec.cam_from_px @ px
  assert isinstance(round_trip_points3d, v3d.Point3d)
  assert round_trip_points3d.shape == spec_shape + (5,)
  dca.testing.assert_allclose(round_trip_points3d, points3d, atol=1e-5)

  # Round-trip with depth=None project to z=1
  px = px.replace(depth=None)
  round_trip_points3d = spec.cam_from_px @ px
  assert isinstance(round_trip_points3d, v3d.Point3d)
  assert round_trip_points3d.shape == spec_shape + (5,)
  np.testing.assert_allclose(
      round_trip_points3d.p[..., -1],
      np.ones(spec_shape + (5,)),
      # atol=1e-4,
  )


def _broadcast_to(xnp: enp.NpModule, array, shape):
  return xnp.broadcast_to(xnp.asarray(array), shape)

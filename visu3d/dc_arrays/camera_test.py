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

"""Tests for camera."""

from __future__ import annotations

from etils import enp
import numpy as np
import pytest
import visu3d as v3d

# Activate the fixture
set_tnp = enp.testing.set_tnp


H, W = 64, 128


def _make_cam(
    *,
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
) -> v3d.Camera:
  """Create a camera at (0, 4, 0) looking at the center."""
  spec = v3d.PinholeCamera.from_focal(resolution=(H, W), focal_in_px=34.)
  cam = v3d.Camera.from_look_at(
      spec=spec.as_xnp(xnp),
      pos=[0, 4, 0],  # Camera on the `y` axis
      target=[0, 0, 0],
  )
  cam = cam.broadcast_to(shape)
  return cam


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (3,)])
def test_camera_properties(xnp: enp.NpModule, shape: v3d.typing.Shape):
  cam = _make_cam(xnp=xnp, shape=shape)

  # Properties
  assert cam.shape == shape
  assert cam.spec.shape == shape
  assert cam.world_from_cam.shape == shape
  assert cam.resolution == (H, W)
  assert cam.h == H
  assert cam.w == W
  assert cam.wh == (W, H)
  assert cam.hw == (H, W)


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (3,), (1, 1)])
@pytest.mark.parametrize('normalize', [False, True])
def test_camera_rays(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
    normalize: bool,
):
  if shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  cam = _make_cam(xnp=xnp, shape=shape)  # Camera on the `y` axis

  rays = cam.rays(normalize=normalize)
  assert isinstance(rays, v3d.Ray)
  assert rays.xnp is xnp
  assert rays.shape == shape + (H, W)
  np.testing.assert_allclose(
      rays.pos,
      cam.world_from_cam[..., None, None].broadcast_to(shape + (H, W)).t,
  )

  if normalize:
    np.testing.assert_allclose(rays.norm(), np.ones(shape + (H, W)), atol=1e-6)
    # Ray is normalized
    np.testing.assert_allclose(
        np.linalg.norm(rays.dir, axis=-1),
        np.ones(shape + (H, W)),
        atol=1e-6,
    )
  else:
    # Ray destinations are aligned with the y=3 plane
    np.testing.assert_allclose(rays.end[..., 1], np.full(shape + (H, W), 3.))

  v3d.testing.assert_array_equal(cam + [0, 0, 0], cam)
  v3d.testing.assert_array_equal(cam + xnp.array([0, 0, 0]), cam)

  _ = cam.fig

  cam = cam.replace_fig_config(scale=3.)
  _ = cam.fig


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (3,), (2, 2)])
def test_camera_transform(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
):
  """Test low-level camera transforms."""

  if shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  cam = _make_cam(xnp=xnp, shape=shape)  # Camera on the `y` axis
  rays = cam.rays(normalize=False)

  with enp.lazy.jax.checking_leaks():
    # px -> cam -> world coordinates
    px = cam.spec.px_centers()
    pts0 = cam.world_from_cam @ (cam.spec.cam_from_px @ px)
    pts1 = (cam.world_from_cam @ cam.spec.cam_from_px) @ px
    pts2 = cam.world_from_px @ px
    np.testing.assert_allclose(pts0, pts1, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(pts0, pts2, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(rays.end, pts0, atol=1e-5, rtol=1e-6)

    # world -> cam -> px coordinates
    px0 = cam.spec.px_from_cam @ (cam.cam_from_world @ pts0)
    px1 = (cam.spec.px_from_cam @ cam.cam_from_world) @ pts0
    px2 = cam.px_from_world @ pts0
    np.testing.assert_allclose(px0, px, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(px1, px, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(px2, px, atol=1e-5, rtol=1e-6)


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize('shape', [(), (3,), (1, 1)])
@pytest.mark.parametrize('point_shape', [(), (3,), (1, 1)])
def test_camera_render(
    xnp: enp.NpModule,
    shape: v3d.typing.Shape,
    point_shape: v3d.typing.Shape,
):
  if shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  cam = _make_cam(xnp=xnp, shape=shape)  # Camera on the `y` axis
  points = v3d.Point3d(
      p=[0, 0, 0],
      rgb=[255., 255., 255.],
  )
  points = points.as_xnp(xnp)
  points = points.broadcast_to((1,) * len(shape) + point_shape)
  if shape and xnp is enp.lazy.jnp:
    pytest.skip('vmap not (yet) supported for cam.render')
  img = cam.render(points)
  assert img.shape == shape + (H, W, 3)
  assert img.dtype == np.uint8
  first_idx = (0,) * len(shape)
  np.testing.assert_allclose(img[(*first_idx, 0, 0)], [0, 0, 0])
  np.testing.assert_allclose(img[(*first_idx, H // 2, W // 2)], [255, 255, 255])

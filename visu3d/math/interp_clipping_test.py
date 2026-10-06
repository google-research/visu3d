# Copyright 2026 The visu3d Authors.
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

"""Regression tests for finite coordinates outside image boundaries."""

import unittest

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import visu3d as v3d


class InterpolationClippingTest(parameterized.TestCase):

  @parameterized.product(
      backend=('numpy', 'jax'),
      hw_coords=(False, True),
      size=((3, 4), (1, 4), (3, 1), (1, 1)),
  )
  def test_far_coordinates_replicate_edge_pixels(
      self, backend, hw_coords, size
  ):
    xnp = np if backend == 'numpy' else jnp
    height, width = size
    image = xnp.asarray(
        np.arange(height * width * 2).reshape(height, width, 2) + 5,
        dtype=xnp.float32,
    )
    coords = np.array(
        [
            [1e8, 1.5],
            [-1e8, 1.5],
            [1.5, 1e10],
            [1.5, -1e10],
            [1e30, 1e30],
            [-1e30, -1e30],
        ],
        dtype=np.float32,
    )
    if hw_coords:
      coords = coords[:, ::-1].copy()
    clipped = coords.copy()
    lower = [0.5, 0.5]
    upper = (
        [height - 0.5, width - 0.5]
        if hw_coords
        else [width - 0.5, height - 0.5]
    )
    clipped = np.clip(clipped, lower, upper).astype(np.float32)
    expected = v3d.math.interp_img(
        image, xnp.asarray(clipped), use_hw_coords=hw_coords
    )
    actual = v3d.math.interp_img(
        image, xnp.asarray(coords), use_hw_coords=hw_coords
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    self.assertTrue(np.isfinite(actual).all())

  def test_interior_weights_and_query_shapes_remain_unchanged(self):
    image = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
    coords = np.array(
        [[[0.5, 0.5], [1.0, 1.0]], [[2.0, 1.5], [4.0, 3.0]]], dtype=np.float32
    )
    actual = v3d.math.interp_img(image, coords)
    np.testing.assert_allclose(actual[..., 0], [[0.0, 2.5], [5.5, 11.0]])
    self.assertEqual(actual.shape, (2, 2, 1))
    empty = v3d.math.interp_img(image, np.empty((0, 2), dtype=np.float32))
    self.assertEqual(empty.shape, (0, 1))

  def test_jitted_image_and_coordinate_gradients_match_edge_extension(self):
    image = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4, 1)
    coords = jnp.array([[1e10, 1.5]], dtype=jnp.float32)

    def loss(img, points):
      return v3d.math.interp_img(img, points).sum()

    value, (image_grad, coord_grad) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1))
    )(image, coords)
    self.assertEqual(float(value), 7.0)
    expected_grad = np.zeros((3, 4, 1))
    expected_grad[1, 3, 0] = 1.0
    np.testing.assert_array_equal(image_grad, expected_grad)
    np.testing.assert_allclose(coord_grad, [[0.0, 4.0]])

  def test_pixel_center_coordinate_gradients_keep_the_original_convention(self):
    image = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4, 1)
    coords = jnp.array([[0.5, 1.5], [1.5, 0.5], [3.5, 2.5]])
    gradient = jax.grad(lambda p: v3d.math.interp_img(image, p).sum())(coords)
    np.testing.assert_allclose(gradient, [[1.0, 4.0], [1.0, 4.0], [0.0, 0.0]])

  def test_projected_near_plane_points_can_sample_image_boundaries(self):
    camera = v3d.PinholeCamera.from_focal(resolution=(3, 4), focal_in_px=10.0)
    point = v3d.Point3d(p=np.array([1000.0, 0.0, 1e-6], dtype=np.float32))
    projected = camera.px_from_cam @ point
    image = np.arange(12.0, dtype=np.float32).reshape(3, 4, 1)
    actual = v3d.math.interp_img(image, projected.p)
    self.assertGreater(float(projected.p[0]), 1e8)
    # At the right edge this image is the linear function 3 + 4 * (y - 0.5).
    expected = 3.0 + 4.0 * (np.clip(projected.p[1], 0.5, 2.5) - 0.5)
    np.testing.assert_allclose(actual, [expected], rtol=1e-6)


if __name__ == '__main__':
  unittest.main()

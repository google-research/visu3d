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

"""Tests."""

from etils import enp
import numpy as np
import pytest
import visu3d as v3d

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize(
    'value, shape',
    [
        (
            [
                [2.0, 4, 0.3],
                [-2.0, -4, -0.3],
            ],
            (2,),
        ),
        (
            [2.0, 4, 0.3],
            (),
        ),
    ],
)
def test_spherical_coordinates(xnp: enp.NpModule, value, shape):
  pts = xnp.asarray(value)

  spherical = v3d.math.carthesian_to_spherical(pts)

  assert enp.lazy.get_xnp(spherical.r) is xnp
  assert enp.lazy.get_xnp(spherical.theta) is xnp
  assert enp.lazy.get_xnp(spherical.phi) is xnp
  assert spherical.r.shape == shape
  assert spherical.theta.shape == shape
  assert spherical.phi.shape == shape

  pts_round_trip = v3d.math.spherical_to_carthesian(*spherical)
  assert enp.lazy.get_xnp(pts_round_trip) is xnp
  assert pts_round_trip.shape == shape + (3,)

  np.testing.assert_allclose(pts, pts_round_trip, atol=1e-6)


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize(
    'point',
    [
        [1e-4, 0.0, 1.0],
        [1e-4, 0.0, -1.0],
        [-1e-4, 2e-4, 1.0],
        [-1e-4, 2e-4, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 2.0, 0.0],
    ],
)
def test_spherical_coordinates_near_poles(xnp: enp.NpModule, point):
  pts = xnp.asarray(np.array(point, dtype=np.float32))
  spherical = v3d.math.carthesian_to_spherical(pts)
  expected_phi = np.arctan2(np.hypot(point[0], point[1]), point[2])

  np.testing.assert_allclose(spherical.phi, expected_phi, rtol=1e-6, atol=1e-7)
  reconstructed = v3d.math.spherical_to_carthesian(*spherical)
  np.testing.assert_allclose(reconstructed, point, rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize('z', [1.0, -1.0])
def test_spherical_polar_angle_gradient_near_poles(z):
  jax = enp.lazy.jax
  point = np.array([1e-4, 2e-4, z], dtype=np.float32)
  grad_phi = jax.jit(
      jax.grad(lambda p: v3d.math.carthesian_to_spherical(p).phi)
  )(point)
  x, y, z = point.astype(np.float64)
  rho = np.hypot(x, y)
  r_squared = rho**2 + z**2
  expected = np.array([x * z / rho, y * z / rho, -rho]) / r_squared

  assert np.isfinite(grad_phi).all()
  np.testing.assert_allclose(grad_phi, expected, rtol=1e-6, atol=1e-7)

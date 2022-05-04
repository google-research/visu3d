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

"""Tests for rotation_utils."""

from etils import enp
import numpy as np
import pytest
from visu3d.utils import rotation_utils

# Activate the fixture
set_tnp = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_is_rotation_matrix(xnp: enp.NpModule):
  assert rotation_utils.is_rot(xnp.eye(3))
  assert not rotation_utils.is_rot(xnp.zeros((3, 3)))

  not_rot = xnp.array([
      [0., 0., 1.],
      [0., 1., 0.],
      [0., 0., 1.],
  ])
  assert not rotation_utils.is_rot(not_rot)

  delta = 1e-8
  rot = xnp.array([
      [1. + delta, 0., 0.],
      [0., 1., 0.],
      [0., 0., 1.],
  ])
  assert rotation_utils.is_rot(rot, atol=delta * 10)


@enp.testing.parametrize_xnp()
def test_is_rotation_matrix_raises(xnp: enp.NpModule):
  with pytest.raises(ValueError, match='Expected 3x3'):
    rotation_utils.is_rot(xnp.eye(2))


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize('rad', [
    0.0,
    1 / 4 * enp.tau,
    -1 / 4 * enp.tau,
    1 / 2 * enp.tau,
    -1 / 2 * enp.tau,
    enp.tau,
])
def test_rotation_around_axis(xnp: enp.NpModule, rad: float):
  if xnp is not None:
    rad = xnp.asarray(rad)
  else:
    xnp = enp.lazy.np
  rx = rotation_utils.rot_x(rad)
  ry = rotation_utils.rot_y(rad)
  rz = rotation_utils.rot_z(rad)
  assert rotation_utils.is_rot(rx)
  assert rotation_utils.is_rot(ry)
  assert rotation_utils.is_rot(rz)
  assert enp.lazy.get_xnp(rx) is xnp
  assert enp.lazy.get_xnp(ry) is xnp
  assert enp.lazy.get_xnp(rz) is xnp

  identity = xnp.eye(3)
  np.testing.assert_allclose(rx @ rotation_utils.rot_x(-rad), identity)
  np.testing.assert_allclose(ry @ rotation_utils.rot_y(-rad), identity)
  np.testing.assert_allclose(rz @ rotation_utils.rot_z(-rad), identity)

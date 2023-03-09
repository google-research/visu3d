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

from __future__ import annotations

from etils import enp
import numpy as np
import pytest
import visu3d as v3d

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_is_rotation_matrix(xnp: enp.NpModule):
  assert v3d.math.is_rot(xnp.eye(3))
  assert not v3d.math.is_rot(xnp.zeros((3, 3)))

  not_rot = xnp.asarray(
      [
          [0.0, 0.0, 1.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
      ]
  )
  assert not v3d.math.is_rot(not_rot)

  delta = 1e-8
  rot = xnp.asarray(
      [
          [1.0 + delta, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
      ]
  )
  assert v3d.math.is_rot(rot, atol=delta * 10)


@enp.testing.parametrize_xnp()
def test_is_rotation_matrix_raises(xnp: enp.NpModule):
  with pytest.raises(ValueError, match='Expected 3x3'):
    v3d.math.is_rot(xnp.eye(2))


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize(
    'rad',
    [
        0.0,
        1 / 4 * enp.tau,
        -1 / 4 * enp.tau,
        1 / 2 * enp.tau,
        -1 / 2 * enp.tau,
        enp.tau,
        # Same shifted by 1 tau
        5 / 4 * enp.tau,
        -5 / 4 * enp.tau,
        3 / 2 * enp.tau,
        -3 / 2 * enp.tau,
        2 * enp.tau,
    ],
)
def test_rotation_around_axis(xnp: enp.NpModule, rad: float):
  if xnp is not None:
    rad = xnp.asarray(rad)
  else:
    xnp = enp.lazy.np
  rx = v3d.math.rot_x(rad)
  ry = v3d.math.rot_y(rad)
  rz = v3d.math.rot_z(rad)
  assert v3d.math.is_rot(rx)
  assert v3d.math.is_rot(ry)
  assert v3d.math.is_rot(rz)
  assert enp.lazy.get_xnp(rx) is xnp
  assert enp.lazy.get_xnp(ry) is xnp
  assert enp.lazy.get_xnp(rz) is xnp

  identity = xnp.eye(3)
  np.testing.assert_allclose(rx @ v3d.math.rot_x(-rad), identity, atol=1e-6)
  np.testing.assert_allclose(ry @ v3d.math.rot_y(-rad), identity, atol=1e-6)
  np.testing.assert_allclose(rz @ v3d.math.rot_z(-rad), identity, atol=1e-6)

  # Round trip euler <> matrix
  _assert_euler_round_trip(rx, xnp=xnp)
  _assert_euler_round_trip(ry, xnp=xnp)
  _assert_euler_round_trip(rz, xnp=xnp)


@enp.testing.parametrize_xnp(with_none=True)
@pytest.mark.parametrize(
    'rx, ry, rz',
    [
        (0.0, 0.0, 0.0),
        (1 / 4, -1 / 3, 1 / 5),
        (5 / 4, -4 / 3, 6 / 5),
    ],
)
def test_euler_roundtrip(xnp: enp.NpModule, rx: float, ry: float, rz: float):
  if xnp is not None:
    rx = xnp.asarray(rx)
    ry = xnp.asarray(ry)
    rz = xnp.asarray(rz)
  else:
    xnp = enp.lazy.np

  rot = v3d.math.euler_to_rot(x=rx, y=ry, z=rz)
  ax, ay, az = v3d.math.rot_to_euler(rot)
  rot_v2 = v3d.math.euler_to_rot(x=ax, y=ay, z=az)

  assert enp.lazy.get_xnp(rot) is xnp
  assert enp.lazy.get_xnp(rot_v2) is xnp
  np.testing.assert_allclose(rot, rot_v2, atol=1e-6, rtol=1e-7)


def _assert_euler_round_trip(rot, xnp):
  ax, ay, az = v3d.math.rot_to_euler(rot)
  assert enp.lazy.get_xnp(ax) is xnp
  assert enp.lazy.get_xnp(ay) is xnp
  assert enp.lazy.get_xnp(az) is xnp
  rot_v2 = v3d.math.euler_to_rot(x=ax, y=ay, z=az)
  assert enp.lazy.get_xnp(rot_v2) is xnp
  np.testing.assert_allclose(
      rot,
      rot_v2,
      atol=1e-6,
      rtol=1e-7,
      err_msg=f'For angles {(ax, ay, az)}',
  )

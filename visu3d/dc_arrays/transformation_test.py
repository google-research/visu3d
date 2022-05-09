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

"""Tests for transformation."""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

from etils import enp
from etils.array_types import FloatArray
import numpy as np
import pytest
import visu3d as v3d

# Activate the fixture
set_tnp = enp.testing.set_tnp


@dataclasses.dataclass
class TransformExpectedValue:
  """Tests values."""
  # Expected rays values after transformation
  expected_pos: FloatArray[..., 3]
  expected_dir: FloatArray[..., 3]

  # Expected transformation values after composition with other tr
  expected_r: FloatArray[..., 3, 3]
  expected_t: FloatArray[..., 3]

  # Transformation params
  R: Optional[FloatArray[3, 3]] = None  # pylint: disable=invalid-name
  t: Optional[FloatArray[3]] = None


# Transformation values
_RAY_POS = np.array([1, 3, 5])
_RAY_DIR = np.array([2, 1, 4])
_TR_R = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
])
_TR_T = np.array([4, 3, 7])

_TR_EXPECTED_VALUES = {
    # Test identity
    'default_args': TransformExpectedValue(
        # Identity should be a no-op
        expected_pos=_RAY_POS,
        expected_dir=_RAY_DIR,
        expected_r=_TR_R,
        expected_t=_TR_T,
    ),
    # Test identity
    'identity': TransformExpectedValue(
        R=np.eye(3),
        t=np.zeros((3,)),
        # Identity should be a no-op
        expected_pos=_RAY_POS,
        expected_dir=_RAY_DIR,
        expected_r=_TR_R,
        expected_t=_TR_T,
    ),
    # Test translation only
    'trans_only': TransformExpectedValue(
        R=np.eye(3),
        t=[3, -1, 2],
        # Only position translated
        expected_pos=_RAY_POS + [3, -1, 2],
        expected_dir=_RAY_DIR,
        expected_r=_TR_R,
        expected_t=_TR_T + [3, -1, 2],
    ),
    # Test translation only
    'trans_only_default_arg': TransformExpectedValue(
        t=[3, -1, 2],
        # Only position translated
        expected_pos=_RAY_POS + [3, -1, 2],
        expected_dir=_RAY_DIR,
        expected_r=_TR_R,
        expected_t=_TR_T + [3, -1, 2],
    ),
    # Test rotation only
    'rot_only': TransformExpectedValue(
        R=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        t=[0, 0, 0],
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_pos=[5, 1, 3],
        expected_dir=[4, 2, 1],
        # Rotation invert axis `(1, 2, 3)` -> `(2, 1, 3)`
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_r=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        expected_t=[7, 4, 3],
    ),
    # Test rotation only
    'rot_only_default_arg': TransformExpectedValue(
        R=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_pos=[5, 1, 3],
        expected_dir=[4, 2, 1],
        # Rotation invert axis `(1, 2, 3)` -> `(2, 1, 3)`
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_r=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        expected_t=[7, 4, 3],
    ),
    # Test translation + rotation
    'rot_trans': TransformExpectedValue(
        R=[
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        t=[3, -1, 2],
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        # And translation applied
        expected_pos=np.array([5, 1, 3]) + [3, -1, 2],
        expected_dir=[4, 2, 1],
        # Rotation invert axis `(1, 2, 3)` -> `(2, 1, 3)`
        # Rotation invert axis `(1, 2, 3)` -> `(3, 1, 2)`
        expected_r=[
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        expected_t=np.array([7, 4, 3]) + [3, -1, 2],
    ),
}


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize(
    'tr_shape, other_shape, expected_shape',
    [
        ((), (), ()),
        ((3,), (3,), (3,)),
        ((3,), (3, 4, 5), (3, 4, 5)),
        ((3,), (1, 4, 5), (3, 4, 5)),
        ((1,), (3, 4, 5), (3, 4, 5)),
    ],
)
@pytest.mark.parametrize(
    'test_values',
    _TR_EXPECTED_VALUES.values(),
    ids=_TR_EXPECTED_VALUES.keys(),
)
def test_transformation(
    xnp: enp.NpModule,
    tr_shape: v3d.typing.Shape,
    other_shape: v3d.typing.Shape,
    expected_shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  init_kwargs = {}
  if test_values.R is not None:
    init_kwargs['R'] = xnp.array(test_values.R)
  if test_values.t is not None:
    init_kwargs['t'] = xnp.array(test_values.t)
  tr = v3d.Transform(**init_kwargs)

  if test_values.R is None and test_values.t is None:
    tr = tr.as_xnp(xnp)
  tr = tr.broadcast_to(tr_shape)

  if tr_shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  _assert_tr_common(tr, tr_shape=tr_shape)

  _assert_ray_transformed(
      tr,
      expected_shape=expected_shape,
      other_shape=other_shape,
      test_values=test_values,
  )

  _assert_point_transformed(
      tr,
      expected_shape=expected_shape,
      other_shape=other_shape,
      test_values=test_values,
  )

  _assert_tr_transformed(
      tr,
      expected_shape=expected_shape,
      other_shape=other_shape,
      test_values=test_values,
  )

  _assert_camera_transformed(
      tr,
      expected_shape=expected_shape,
      other_shape=other_shape,
      test_values=test_values,
  )


def _assert_tr_common(tr: v3d.Transform, tr_shape: v3d.typing.Shape):
  """Generic rules applied to all transformation."""
  assert tr.shape == tr_shape
  assert tr.R.shape == tr_shape + (3, 3)
  assert tr.t.shape == tr_shape + (3,)

  identity_tr = v3d.Transform(R=tr.xnp.eye(3), t=tr.xnp.zeros((3,)))
  v3d.testing.assert_array_equal(
      identity_tr,
      v3d.Transform.identity().as_xnp(tr.xnp),
  )
  identity_tr = identity_tr.broadcast_to(tr_shape)
  assert identity_tr.shape == tr_shape
  assert identity_tr.R.shape == tr_shape + (3, 3)
  assert identity_tr.t.shape == tr_shape + (3,)

  # Inverting the matrix is equivalent to the matrix of the invert transform
  v3d.testing.assert_array_equal(tr.inv.matrix4x4, enp.linalg.inv(tr.matrix4x4))

  # Inverting twice the transformation should be a no-op
  v3d.testing.assert_array_equal(tr.inv.inv, tr)

  # Composing the transformation with the inverse should be identity
  v3d.testing.assert_array_equal(tr.inv @ tr, identity_tr)
  v3d.testing.assert_array_equal(tr @ tr.inv, identity_tr)
  v3d.testing.assert_array_equal(tr @ identity_tr, tr)
  v3d.testing.assert_array_equal(identity_tr @ tr, tr)

  # Exporting/importing matrix from 4x4 should be a no-op
  v3d.testing.assert_array_equal(v3d.Transform.from_matrix(tr.matrix4x4), tr)

  v3d.testing.assert_array_equal(tr + [0, 0, 0], tr)
  v3d.testing.assert_array_equal(tr + tr.xnp.array([0, 0, 0]), tr)

  # Figure should work
  _ = tr.fig


def _assert_ray_transformed(
    tr: v3d.Transform,
    other_shape: v3d.typing.Shape,
    expected_shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  """Test ray transformation."""
  xnp = tr.xnp
  ray = v3d.Ray(
      pos=xnp.array(_RAY_POS),
      dir=xnp.array(_RAY_DIR),
  )
  ray = ray.broadcast_to(other_shape)
  assert ray.shape == other_shape

  expected_ray = v3d.Ray(
      pos=xnp.array(test_values.expected_pos),
      dir=xnp.array(test_values.expected_dir),
  )
  expected_ray = expected_ray.broadcast_to(expected_shape)
  v3d.testing.assert_array_equal(tr @ ray, expected_ray)


def _assert_point_transformed(
    tr: v3d.Transform,
    other_shape: v3d.typing.Shape,
    expected_shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  """Test point transformation."""
  xnp = tr.xnp

  # Test transform point position
  expected_point_pos = xnp.array(test_values.expected_pos)
  expected_point_pos = xnp.broadcast_to(
      expected_point_pos,
      expected_shape + (3,),
  )

  point_pos = xnp.array(_RAY_POS)
  point_pos = xnp.broadcast_to(point_pos, other_shape + (3,))

  v3d.testing.assert_array_equal(tr @ point_pos, expected_point_pos)
  v3d.testing.assert_array_equal(tr.apply_to_pos(point_pos), expected_point_pos)

  # Test transform point direction
  expected_point_dir = xnp.array(test_values.expected_dir)
  expected_point_dir = xnp.broadcast_to(
      expected_point_dir,
      expected_shape + (3,),
  )

  point_dir = xnp.array(_RAY_DIR)
  point_dir = xnp.broadcast_to(point_dir, other_shape + (3,))

  v3d.testing.assert_array_equal(tr.apply_to_dir(point_dir), expected_point_dir)


def _assert_tr_transformed(
    tr: v3d.Transform,
    other_shape: v3d.typing.Shape,
    expected_shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  """Test transform transformation."""
  xnp = tr.xnp
  other_tr = v3d.Transform(
      R=xnp.array(_TR_R),
      t=xnp.array(_TR_T),
  )
  other_tr = other_tr.broadcast_to(other_shape)

  expected_tr = v3d.Transform(
      R=xnp.array(test_values.expected_r),
      t=xnp.array(test_values.expected_t),
  )
  expected_tr = expected_tr.broadcast_to(expected_shape)

  v3d.testing.assert_array_equal(tr @ other_tr, expected_tr)

  # Composing transformations or matrix is equivalent
  # Only test for scalar shape as broadcasting & vectorization have different
  # rules
  if not tr.shape:
    v3d.testing.assert_array_equal(
        v3d.Transform.from_matrix(tr.matrix4x4 @ other_tr.matrix4x4),
        expected_tr,
    )


def _assert_camera_transformed(
    tr: v3d.Transform,
    other_shape: v3d.typing.Shape,
    expected_shape: v3d.typing.Shape,
    test_values: TransformExpectedValue,
):
  """Test transform transformation."""
  xnp = tr.xnp
  spec = v3d.PinholeCamera.from_focal(
      resolution=(12, 12),
      focal_in_px=34,
  )
  spec = spec.as_xnp(xnp)
  cam = v3d.Camera(
      spec=spec,
      world_from_cam=v3d.Transform(
          R=xnp.array(_TR_R),
          t=xnp.array(_TR_T),
      ),
  )
  cam = cam.broadcast_to(other_shape)

  expected_cam = v3d.Camera(
      spec=spec,
      world_from_cam=v3d.Transform(
          R=xnp.array(test_values.expected_r),
          t=xnp.array(test_values.expected_t),
      ),
  )
  expected_cam = expected_cam.broadcast_to(expected_shape)

  v3d.testing.assert_array_equal(tr @ cam, expected_cam)


def _assert_scale(
    tr: v3d.Transform,
    *,
    xnp: enp.NpModule,
    tr_shape: v3d.typing.Shape,
    expected_r: FloatArray[3, 3],
    expected_scale_xyz: FloatArray[3],
    expected_scale: Optional[FloatArray['']],
):
  assert tr.xnp is xnp
  assert tr.shape == tr_shape
  expected_r = xnp.asarray(expected_r)
  expected_scale_xyz = xnp.asarray(expected_scale_xyz)
  np.testing.assert_allclose(
      tr.R,
      xnp.broadcast_to(expected_r, tr_shape + (3, 3)),
      atol=1e-6,
  )
  np.testing.assert_allclose(
      tr.scale_xyz,
      xnp.broadcast_to(expected_scale_xyz, tr_shape + (3,)),
  )
  if expected_scale is None:
    with pytest.raises(ValueError, match='Cannot get `Transform.scale`'):
      _ = tr.scale
  else:
    expected_scale = xnp.asarray(expected_scale)
    np.testing.assert_allclose(
        tr.scale,
        xnp.broadcast_to(expected_scale, tr_shape),
    )


@enp.testing.parametrize_xnp()
@pytest.mark.parametrize(
    'tr_shape',
    [
        (),
        (5,),
    ],
)
def test_transformation_scale(
    xnp: enp.NpModule,
    tr_shape: v3d.typing.Shape,
):
  tr = v3d.Transform.from_angle(x=xnp.asarray(enp.tau / 4))
  tr = tr.broadcast_to(tr_shape)

  if tr_shape and xnp is enp.lazy.tnp:
    pytest.skip('Vectorization not supported yet with TF')

  assert_scale = functools.partial(_assert_scale, xnp=xnp, tr_shape=tr_shape)

  assert_scale(
      tr,
      expected_r=[
          [1, 0, 0],
          [0, 0, -1],
          [0, 1, 0],
      ],
      expected_scale_xyz=[1, 1, 1],
      expected_scale=1,
  )
  assert_scale(
      tr.mul_scale(3.),
      expected_r=[
          [3, 0, 0],
          [0, 0, -3],
          [0, 3, 0],
      ],
      expected_scale_xyz=[3, 3, 3],
      expected_scale=3,
  )
  assert_scale(
      tr.mul_scale(3.).normalize(),
      expected_r=[
          [1, 0, 0],
          [0, 0, -1],
          [0, 1, 0],
      ],
      expected_scale_xyz=[1, 1, 1],
      expected_scale=1,
  )
  if xnp == enp.lazy.jnp and tr_shape:
    # Jax don't support error checking + vectorization, so will return smallest
    expected_scale = 1.5
  else:
    expected_scale = None
  assert_scale(
      tr.mul_scale([2, -3, 1.5]),
      expected_r=[
          [2, 0, 0],
          [0, 0, -1.5],
          [0, -3, 0],
      ],
      # TODO(epot): Detect scale sign
      expected_scale_xyz=[2, 3, 1.5],
      expected_scale=expected_scale,
  )


@enp.testing.parametrize_xnp()
def test_transformation_from_angle_multiple(xnp: enp.NpModule):
  # TODO(epot): Supports vectorization
  z = enp.tau / 4
  y = -enp.tau / 8
  x = xnp.asarray(enp.tau / 16)

  rz = v3d.utils.rotation_utils.rot_z(z)
  ry = v3d.utils.rotation_utils.rot_y(y)
  rx = v3d.utils.rotation_utils.rot_x(x)

  tr = v3d.Transform.from_angle(x=x, y=y, z=z)

  assert tr.xnp is xnp

  v3d.testing.assert_array_equal(tr.R, rz @ ry @ rx)

  v3d.testing.assert_array_equal(
      v3d.Transform.from_angle(),
      v3d.Transform.identity(),
  )


@enp.testing.parametrize_xnp()
def test_transformation_from_angle(xnp: enp.NpModule):
  # TODO(epot): Supports vectorization
  z = xnp.asarray(enp.tau / 4)
  y = xnp.asarray(-enp.tau / 8)
  x = xnp.asarray(enp.tau / 16)

  rz = v3d.utils.rotation_utils.rot_z(z)
  ry = v3d.utils.rotation_utils.rot_y(y)
  rx = v3d.utils.rotation_utils.rot_x(x)

  tr_x = v3d.Transform.from_angle(x=x)
  tr_y = v3d.Transform.from_angle(y=y)
  tr_z = v3d.Transform.from_angle(z=z)
  assert tr_x.xnp is xnp
  assert tr_y.xnp is xnp
  assert tr_z.xnp is xnp

  v3d.testing.assert_array_equal(tr_x.R, rx)
  v3d.testing.assert_array_equal(tr_y.R, ry)
  v3d.testing.assert_array_equal(tr_z.R, rz)

  # Identity
  v3d.testing.assert_array_equal(
      v3d.Transform.from_angle(),
      v3d.Transform.identity(),
  )

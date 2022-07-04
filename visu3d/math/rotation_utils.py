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

"""Rotation utils.

Attributes:
  DEG2RAD: Constant to convert degrees to radians.
  RAD2DEG: Constant to convert radians to degrees.
"""

from __future__ import annotations

from etils import enp
from etils.array_types import FloatArray

DEG2RAD = enp.tau / 360.
RAD2DEG = 360. / enp.tau

# TODO(epot): Support vectorization


# Could eventually add utils to combine rotations together, like:
# yaw == ψ == psi == z
# pitch == θ == theta == y
# roll == ϕ == phi == x
# But might be better to let users explicitly compose transforms.


def rot_x(angle: FloatArray['']) -> FloatArray['3 3']:
  """Rotation matrix for rotation around X (in radians)."""
  xnp = enp.lazy.get_xnp(angle, strict=False)
  angle = xnp.asarray(angle)
  if angle.ndim:
    raise ValueError(f'Rotation angle should be scalar. Not {angle.shape}')
  c = xnp.cos(angle)
  s = xnp.sin(angle)
  R = xnp.array([  # pylint: disable=invalid-name
      [1, 0, 0],
      [0, c, -s],
      [0, s, c],
  ])
  return R


def rot_y(angle: FloatArray['']) -> FloatArray['3 3']:
  """Rotation matrix for rotation around Y (in radians)."""
  xnp = enp.lazy.get_xnp(angle, strict=False)
  angle = xnp.asarray(angle)
  if angle.ndim:
    raise ValueError(f'Rotation angle should be scalar. Not {angle.shape}')
  c = xnp.cos(angle)
  s = xnp.sin(angle)
  R = xnp.array([  # pylint: disable=invalid-name
      [c, 0, s],
      [0, 1, 0],
      [-s, 0, c],
  ])
  return R


def rot_z(angle: FloatArray['']) -> FloatArray['3 3']:
  """Rotation matrix for rotation around Z (in radians)."""
  xnp = enp.lazy.get_xnp(angle, strict=False)
  angle = xnp.asarray(angle)
  if angle.ndim:
    raise ValueError(f'Rotation angle should be scalar. Not {angle.shape}')
  c = xnp.cos(angle)
  s = xnp.sin(angle)
  R = xnp.array([  # pylint: disable=invalid-name
      [c, -s, 0],
      [s, c, 0],
      [0, 0, 1],
  ])
  return R


def euler_to_rot(
    x=None,
    y=None,
    z=None,
    *,
    order: str = 'zyx',
) -> FloatArray['3 3']:
  """Creates a 3x3 matrix from the euler radian angles.

  By default, rotations are applied following the Tait-Bryan chained rotations
  (z, y, x):

  ```python
  R = Rz @ Ry @ Rx
  ```

  See:
  https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_angles_(z-y%E2%80%B2-x%E2%80%B3_intrinsic)_%E2%86%92_rotation_matrix

  Args:
    x: Rotation around x (in radians): roll == ϕ == phi == x
    y: Rotation around y (in radians): pitch == θ == theta == y
    z: Rotation around z (in radians): yaw == ψ == psi == z
    order: Axis order convention used (e.g. `'xyz'`, `'zyx'`,...)

  Returns:
    tr: The transformation.
  """
  order = tuple(order)

  if set(order) != set('xyz') or len(order) != 3:
    raise ValueError(
        f'Order should contain x, y, z exactly once. Got {order!r}')

  def _accumulate_rot(r0, r1):
    return r1 if r0 is None else r0 @ r1

  axis_to_val = {'x': x, 'y': y, 'z': z}
  axis_to_rot_fn = {
      'x': rot_x,
      'y': rot_y,
      'z': rot_z,
  }

  r_final = None
  for axis in order:
    angle = axis_to_val[axis]
    if angle is None:
      continue
    r = axis_to_rot_fn[axis](angle)
    r_final = _accumulate_rot(r_final, r)

  if r_final is None:  # All x, y, z undefined => Identity
    r_final = enp.lazy.np.eye(3)
  return r_final


def rot_to_euler(
    rot: FloatArray['3 3'],
    *,
    eps: float = 1e-6,
) -> tuple[FloatArray[''], FloatArray[''], FloatArray['']]:
  """Extract euler angles from a 3x3 rotation matrix.

  Like `euler_to_rot`, it follow the z, y, x convension, BUT returns x, y, z.

  From: https://www.geometrictools.com/Documentation/EulerAngles.pdf

  Args:
    rot: Rotation matrix
    eps: Precision threshold to detect 90 degree angles.

  Returns:
    The x, y, z euler angle (in radian)
  """
  xnp = enp.lazy.get_xnp(rot)

  r00 = rot[0, 0]
  # r01 = rot[0, 1]
  r02 = rot[0, 2]
  r10 = rot[1, 0]
  r11 = rot[1, 1]
  r12 = rot[1, 2]
  r20 = rot[2, 0]
  r21 = rot[2, 1]
  r22 = rot[2, 2]

  if xnp.abs(r20) < 1. - eps:  # Should allow to tune the precision ?
    theta_y = xnp.arcsin(-r20)
    theta_z = xnp.arctan2(r10, r00)
    theta_x = xnp.arctan2(r21, r22)
  else:  # r20 == +1 / -1
    sign = +1 if r02 > 0 else -1

    theta_y = sign * enp.tau / 4
    theta_z = -sign * xnp.arctan2(-r12, r11)
    theta_x = 0

  theta_x = xnp.asarray(theta_x)
  theta_y = xnp.asarray(theta_y)
  theta_z = xnp.asarray(theta_z)
  return theta_x, theta_y, theta_z


def is_orth(rot: FloatArray['3 3'], *, atol: float = 1e-6) -> bool:
  """Check if the matrix is a valid orthogonal matrix `O(3)`.

  Each orthogonal matrix form a orthonormal basis.
  The group of all 3x3 orthogonal matrices consists of all proper and improper
  rotations (reflexions).

  See:
  https://en.wikipedia.org/wiki/3D_rotation_group#Orthogonal_and_rotation_matrices

  Args:
    rot: The 3x3 matrix to check
    atol: Precision at which checking the matrix

  Returns:
    True if the matrix is orthogonal.
  """
  xnp = enp.lazy.get_xnp(rot)
  if rot.shape != (3, 3):
    raise ValueError(f'Expected 3x3 shape, but got {rot.shape}')

  should_be_identity = rot.T @ rot
  identity = xnp.eye(3, dtype=rot.dtype)
  diff = enp.linalg.norm(identity - should_be_identity)
  return diff < atol


def is_rot(rot: FloatArray['3 3'], *, atol: float = 1e-6) -> bool:
  """Checks if a matrix is a valid rotation matrix `SO(3)`.

  This is done by checking:

  * R.T @ R = Identity (R is orthogonal)
  * det(R) = +1 (-1 corresponding to reflexions)

  If you don't care about reflexions, use `is_orth` instead.

  See:
  https://en.wikipedia.org/wiki/3D_rotation_group#Orthogonal_and_rotation_matrices

  Args:
    rot: The 3x3 matrix to check
    atol: Precision at which checking the matrix

  Returns:
    True if the matrix is a rotation matrix.
  """
  if rot.shape != (3, 3):
    raise ValueError(f'Expected 3x3 shape, but got {rot.shape}')

  det = enp.linalg.det(rot)
  return is_orth(rot, atol=atol) and det > 0.

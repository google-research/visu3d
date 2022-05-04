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

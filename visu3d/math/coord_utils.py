# Copyright 2023 The visu3d Authors.
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

"""Coordinate conversion utils."""

from typing import NamedTuple, Optional

from dataclass_array.typing import FloatArray
from etils import enp


class _SphericalCoords(NamedTuple):
  """(r, theta, phi) == (r, azimuth, elevation).

  Follow https://mathworld.wolfram.com/SphericalCoordinates.html
  conventions.

  Attributes:
    r: radial
    theta: azimuth / longitude `(0, tau)`
    elevation: polar / elevation /  colatitude `(0, tau/2)`
  """

  r: FloatArray['*shape']
  theta: FloatArray['*shape']
  phi: FloatArray['*shape']


@enp.check_and_normalize_arrays(strict=False)
def carthesian_to_spherical(
    point3d: FloatArray['... 3'],
    *,
    xnp: enp.NpModule = ...,
) -> _SphericalCoords:
  """Convert (x, y, z) to (r, theta, phi).

  Follow https://mathworld.wolfram.com/SphericalCoordinates.html
  conventions.

  Args:
    point3d: 3d carthesian coordinates
    xnp: Numpy module

  Returns:
    r: radial
    theta: azimuth / longitude in range `(0, tau)`
    elevation: polar / elevation /  colatitude in range `(0, tau/2)`
  """
  r = enp.compat.norm(point3d, axis=-1)
  theta = xnp.arctan2(point3d[..., 1], point3d[..., 0])  # (-tau/2, tau/2)
  theta = theta % enp.tau  # Normalize azimuth (-tau/2, tau/2) -> (0, tau)
  phi = xnp.arccos(point3d[..., 2] / r)  # elevation (0, tau/2)

  return _SphericalCoords(r, theta, phi)


@enp.check_and_normalize_arrays(strict=False)
def spherical_to_carthesian(
    r: Optional[FloatArray['*shape']] = None,
    theta: Optional[FloatArray['*shape']] = None,
    phi: Optional[FloatArray['*shape']] = None,
    *,
    xnp: enp.NpModule = ...,
) -> FloatArray['*shape 3']:
  """Convert (r, theta, phi) to (x, y, z).

  Follow https://mathworld.wolfram.com/SphericalCoordinates.html
  conventions.

  Usage:

  ```
  v3d.math.spherical_to_carthesian(r, theta, phi)
  v3d.math.spherical_to_carthesian(theta=theta, phi=phi)
  ```

  Args:
    r: If `None`, assume 1.
    theta: azimuth / longitude
    phi: polar / elevation /  colatitude
    xnp: Numpy module

  Returns:
    points: 3d coordinates
  """
  if theta is None or phi is None:
    raise ValueError('theta and phi should be given.')
  points3d = xnp.stack(
      [
          xnp.sin(phi) * xnp.cos(theta),
          xnp.sin(phi) * xnp.sin(theta),
          xnp.cos(phi),
      ],
      axis=-1,
  )
  if r is not None:
    points3d = points3d * r[..., None]
  return points3d

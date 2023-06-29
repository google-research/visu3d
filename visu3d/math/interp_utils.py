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

"""Interpolation utils."""

from etils import enp
from etils.array_types import Array, FloatArray  # pylint: disable=g-multiple-import


@enp.check_and_normalize_arrays(strict=False)
def interp_img(
    img: Array['h w c'],
    coords: FloatArray['*shape 2'],
    *,
    xnp: enp.NpModule = ...,
) -> FloatArray['*shape c']:
  """Bilinear interpolation of coordinates from an image.

  The following assumptions are made:

  * Interpolation is performed from the pixel center (`coords=[0.5, 0.5]`
    match `img[0, 0]`).
  * Pixel coordinates are given in `coords=[w, h]`, NOT `[h, w]`. This is
    consistent with `cam.spec.px_centers()` and OpenCv conventions.
    To invert, use `coords[..., ::-1]`
  * Coordinates outside the images use the image corners value.

  Args:
    img: Image from which interpolating values
    coords: Pixel coordinates from which interpolate (in `w, h` convention).
    xnp: Numpy module to use

  Returns:
    The interpolated pixel values at the requested coordinates.
    Note that returned value is always float (so interpolation of `uint8`
    return float)
  """
  h, w, c = img.shape
  *coord_shape, _ = coords.shape

  # From https://en.wikipedia.org/wiki/Bilinear_interpolation

  # TODO(b/262392450): Should use `.flatten()` instead
  x = coords[..., 0].reshape((-1,))  # w
  y = coords[..., 1].reshape((-1,))  # h

  # For each query coordinates, extract the 4 corners coordinates
  # Pixel coordinates are centered, so keep both `_i` (integer index) and
  # `_f` (float centered coordinates).
  x0 = xnp.floor(x - 0.5)
  x0f = x0 + 0.5
  x0i = enp.compat.astype(x0, xnp.int32)
  x1f = x0f + 1
  x1i = x0i + 1
  y0 = enp.compat.astype(xnp.floor(y - 0.5), xnp.int32)
  y0f = y0 + 0.5
  y0i = enp.compat.astype(y0, xnp.int32)
  y1f = y0f + 1
  y1i = y0i + 1

  del x0, y0  # Should explicitly use integer `_i` or float `_f`

  # Compute the weights for the 4 corners
  # This has to be applied before clipping
  wa = (x1f - x) * (y1f - y)
  wb = (x1f - x) * (y - y0f)
  wc = (x - x0f) * (y1f - y)
  wd = (x - x0f) * (y - y0f)

  # Clip coordinates outside the image
  x0i = xnp.clip(x0i, 0, w - 1)
  x1i = xnp.clip(x1i, 0, w - 1)
  y0i = xnp.clip(y0i, 0, h - 1)
  y1i = xnp.clip(y1i, 0, h - 1)

  if enp.lazy.is_torch_xnp(xnp):
    # Pytorch indexing do not support int32
    x0i = enp.compat.astype(x0i, xnp.int64)
    x1i = enp.compat.astype(x1i, xnp.int64)
    y0i = enp.compat.astype(y0i, xnp.int64)
    y1i = enp.compat.astype(y1i, xnp.int64)

  # Extract the `(num_coords, c)` value for each corners
  val_a = img[y0i, x0i]
  val_b = img[y1i, x0i]
  val_c = img[y0i, x1i]
  val_d = img[y1i, x1i]

  # Weighted sum for each corners
  # `(num_coords, c) * (num_coords, 1) -> (num_coords, c)`
  pts = (
      val_a * wa[..., None]
      + val_b * wb[..., None]
      + val_c * wc[..., None]
      + val_d * wd[..., None]
  )

  assert pts.shape[-1] == c
  return pts.reshape(coord_shape + [c])

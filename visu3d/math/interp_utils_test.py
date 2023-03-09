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

"""Test."""

from etils import enp
import numpy as np
import visu3d as v3d

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_interp_img(xnp: enp.NpModule):
  img = xnp.asarray([
      [0, 1, 2, 3],
      [1, 11, 21, 31],
      [2, 12, 22, 32],
  ])
  out = v3d.math.interp_img(
      img[..., None],
      coords=[
          [0.5, 0.5],  # Top-left corner
          [2.5, 1.5],  # Inside
          [2.0, 0.5],  # Between 2 cells
          [5.0, 1.5],  # Out of boundaries
      ],
  )
  assert out.shape == (4, 1)
  assert enp.lazy.get_xnp(out) is xnp

  np.testing.assert_allclose(
      out,
      [
          [0],
          [21],
          [1.5],
          [31],
      ],
      atol=1e-6,
  )


@enp.testing.parametrize_xnp()
def test_interp_img_center(xnp: enp.NpModule):
  # Interpolating an image using the pixel coordinates is a no-op
  h, w = 12, 30
  spec = v3d.PinholeCamera.from_focal(resolution=(h, w), focal_in_px=34.0)
  spec = spec.as_xnp(xnp)

  rng = np.random.default_rng(0)
  img = rng.random((h, w, 3))
  centers = spec.px_centers()
  out = v3d.math.interp_img(img, coords=centers)
  assert out.shape == (h, w, 3)
  assert enp.lazy.get_xnp(out) is xnp

  # Sampling the pixel center should be a no-op
  np.testing.assert_allclose(out, img, atol=1e-6)

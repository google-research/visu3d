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

"""Tests for np_utils."""

from __future__ import annotations

from etils import enp
import numpy as np
import visu3d as v3d
from visu3d.utils import np_utils

# Activate the fixture
enable_tf_np_mode = enp.testing.set_tnp


@enp.testing.parametrize_xnp()
def test_append_row(xnp: enp.NpModule):
  x = xnp.ones((2, 4))
  y = np_utils.append_row(x, value=4.0, axis=-1)
  assert enp.compat.is_array_xnp(y, xnp)
  expected = [
      [1, 1, 1, 1, 4],
      [1, 1, 1, 1, 4],
  ]
  np.testing.assert_allclose(y, expected)

  y = np_utils.append_row(x, value=4.0, axis=0)
  assert enp.compat.is_array_xnp(y, xnp)
  expected = [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [4, 4, 4, 4],
  ]
  np.testing.assert_allclose(y, expected)


def test_interpolate():
  p = [
      [0, 0, 0],
      [0, 2, 0],
      [0, 2, 1],
      [0, 0, -5],
  ]
  p_interp = v3d.math.interp_points(p, t=100)
  assert p_interp.shape == (100, 3)

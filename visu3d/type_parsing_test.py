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

"""Tests for type_parsing."""

from __future__ import annotations

from typing import Optional, List, Union

from etils.array_types import f32, FloatArray  # pylint: disable=g-multiple-import
import pytest
import visu3d as v3d
from visu3d import type_parsing


@pytest.mark.parametrize(
    'hint, expected',
    [
        (int, [int]),
        (v3d.Ray, [v3d.Ray]),
        (Union[v3d.Ray, int], [v3d.Ray, int]),
        (Union[v3d.Ray, int, None], [v3d.Ray, int, None]),
        (Optional[v3d.Ray], [v3d.Ray, None]),
        (Optional[Union[v3d.Ray, int]], [v3d.Ray, int, None]),
        (List[int], [List[int]]),
        (f32[3, 3], [f32[3, 3]]),
    ],
)
def test_get_leaf_types(hint, expected):
  assert type_parsing._get_leaf_types(hint) == expected


@pytest.mark.parametrize(
    'hint, expected',
    [
        (int, None),
        (v3d.Ray, v3d.Ray),
        (Optional[v3d.Ray], v3d.Ray),
        (Union[v3d.Ray, v3d.Camera], v3d.DataclassArray),
        (Union[v3d.Ray, v3d.Camera, None], v3d.DataclassArray),
        (Union[v3d.Ray, int], None),
        (Union[v3d.Ray, int, None], None),
        (Union[f32[3, 3], int, None], None),
        (List[int], None),
        (List[v3d.Ray], None),
        (f32[3, 3], f32[3, 3]),
        (FloatArray[..., 3], FloatArray[..., 3]),
    ],
)
def test_get_array_type(hint, expected):
  assert type_parsing.get_array_type(hint) == expected


def test_get_array_type_error():
  with pytest.raises(NotImplementedError):
    type_parsing.get_array_type(Union[v3d.Ray, f32[3, 3]])

  with pytest.raises(NotImplementedError):
    type_parsing.get_array_type(Union[FloatArray[..., 3], f32[3, 3]])

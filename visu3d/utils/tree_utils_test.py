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

"""Tests for tree_utils."""

from __future__ import annotations

import chex
from visu3d.utils import tree_utils


@chex.dataclass
class A:
  x: int
  y: int


def test_tree_map():
  """Test the mapping function."""

  def map_fn(x):
    return x * 10

  result = tree_utils.tree_map(map_fn, {
      'a': 1,
      'b': {
          'c': 2,
          'e': [3, 4, 5],
      },
  })
  assert result == {
      'a': 10,
      'b': {
          'c': 20,
          'e': [30, 40, 50],
      },
  }

  assert tree_utils.tree_map(map_fn, [1, 2, 3]) == [10, 20, 30]
  assert tree_utils.tree_map(map_fn, (1, 2, 3)) == (10, 20, 30)
  assert tree_utils.tree_map(map_fn, {}) == {}  # pylint: disable=g-explicit-bool-comparison
  assert tree_utils.tree_map(map_fn, ()) == ()  # pylint: disable=g-explicit-bool-comparison
  assert tree_utils.tree_map(map_fn, {'x': ([])}) == {'x': ([])}
  assert tree_utils.tree_map(map_fn, 1) == 10


def test_tree_map_multi_args():
  """Test the mapping function."""

  def add_fn(x, y):
    return x + y

  x0 = {'a': [1, 2]}
  x1 = {'a': [10, 20]}
  assert tree_utils.tree_map(add_fn, x0, x1) == {'a': [11, 22]}


def test_tree_map_chex():
  """Test tree_map with chex dataclasses."""

  assert tree_utils.tree_map(lambda x: x * 10, A(x=1, y=2)) == A(x=10, y=20)

  def add_fn(x, y):
    return x + y

  x0 = {'a': A(x=1, y=2)}
  x1 = {'a': A(x=10, y=20)}
  assert tree_utils.tree_map(add_fn, x0, x1) == {'a': A(x=11, y=22)}

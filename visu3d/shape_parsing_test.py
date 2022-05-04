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

"""Tests for shape_parsing."""

from __future__ import annotations

from etils.array_types import f32
import pytest
from visu3d import shape_parsing


_VarDim = shape_parsing._VarDim
_NamedDim = shape_parsing._NamedDim


@pytest.mark.parametrize(
    'shape_str, shape_tuple',
    [
        ('', ()),
        ('_ 3 d', (None, 3, _NamedDim('d'))),
        ('...', (_VarDim('_'),)),
        ('... 3', (_VarDim('_'), 3)),
        ('... 3 5', (_VarDim('_'), 3, 5)),
        (
            '... 3 d 7 other',
            (_VarDim('_'), 3, _NamedDim('d'), 7, _NamedDim('other')),
        ),
        ('*shape', (_VarDim('shape'),)),
        ('*shape 3', (_VarDim('shape'), 3)),
        (
            '3 _ *x *x2 7 some_dim',
            (3, None, _VarDim('x'), _VarDim('x2'), 7, _NamedDim('some_dim')),
        ),
    ],
)
def test_parse_shape(shape_str, shape_tuple):
  parser = shape_parsing.ShapeParser.singleton()
  assert parser.parse(shape_str) == shape_tuple
  assert parser.parse(f32[shape_str].shape) == shape_tuple


def test_parse_shape_types():
  parser = shape_parsing.ShapeParser.singleton()
  assert parser.parse(f32[2, 3].shape) == (2, 3)
  assert parser.parse(f32[..., 3].shape) == (_VarDim('_'), 3)
  assert parser.parse(f32[None, 3, 'd'].shape) == (None, 3, _NamedDim('d'))


@pytest.mark.parametrize(
    'shape_str, shape_tuple',
    [
        ('...', ()),
        ('... 3', (3,)),
        ('... 3 5', (3, 5)),
        ('... 3 5 7', (3, 5, 7)),
        ('*shape', ()),
        ('*shape 3', (3,)),
        ('*shape 3 5', (3, 5)),
        ('*shape 3 5 7', (3, 5, 7)),
    ],
)
def test_get_inner_shape(shape_str, shape_tuple):
  assert shape_parsing.get_inner_shape(shape_str) == shape_tuple


@pytest.mark.parametrize(
    'shape_str',
    [
        '',
        '1 2',
    ],
)
def test_get_inner_shape_failure_first_dim(shape_str: str):
  with pytest.raises(ValueError, match='Shape should start'):
    shape_parsing.get_inner_shape(shape_str)


@pytest.mark.parametrize(
    'shape_str',
    [
        '... ...',
        '... _',
        '... 3 d 1',
        '*shape _',
        '*shape 3 d 1',
    ],
)
def test_get_inner_shape_failure_dynamic(shape_str: str):
  with pytest.raises(ValueError, match='Only static dimension'):
    shape_parsing.get_inner_shape(shape_str)

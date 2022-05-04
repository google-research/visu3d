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

"""Tree utils.

Because `jax.tree_utils`, `tf.nest`, ... recurse inside `v3d.DataclassArray`,
we need another API which doesn't.

"""

from __future__ import annotations

import collections.abc
from typing import Callable, TypeVar

from etils import epy
from etils.etree.typing import Tree
from typing_extensions import Unpack, TypeVarTuple  # pytype: disable=not-supported-yet  # pylint: disable=g-multiple-import


_InsT = TypeVarTuple('_InsT')
_OutT = TypeVar('_OutT')


def tree_map(  # pylint: disable=redefined-builtin
    fn: Callable[[Unpack[Tree[_InsT]]], _OutT],
    *trees: Unpack[Tree[_InsT]],
) -> Tree[_OutT]:
  """Apply a function recursively to each element of a nested data struct.

  Contrary to `jax.tree_utils`, `v3d.DataclassArray` are leaf when used with
  this function.

  Args:
    fn: Function to map.
    *trees: Args to pass tht fn. Should have the same structure.

  Returns:
    Same structure as `trees` after `fn` has been applied.
  """
  arg = trees[0]

  for struct_cls, map_fn in _TYPE_TO_MAP_FN.items():
    if isinstance(arg, struct_cls):
      return map_fn(fn, *trees)
  # Leaf
  return fn(*trees)


def _map_mapping(fn, *dicts):
  dict_cls = type(dicts[0])
  return dict_cls((k, tree_map(fn, *vals)) for k, vals in epy.zip_dict(*dicts))


def _map_sequence(fn, *lists):
  list_cls = type(lists[0])
  # TODO(py3.10): Use strict=True
  return list_cls(tree_map(fn, *vals) for vals in zip(*lists))


_TYPE_TO_MAP_FN = {
    dict: _map_mapping,
    collections.abc.Mapping: _map_mapping,
    tuple: _map_sequence,
    list: _map_sequence,
    collections.abc.Sequence: _map_mapping,
}

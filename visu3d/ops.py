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

"""Operation utils."""

from __future__ import annotations

from typing import Iterable  # pylint: disable=g-multiple-import

from visu3d import array_dataclass
from visu3d.typing import DcT
from visu3d.utils import np_utils
from visu3d.utils import py_utils


def stack(
    arrays: Iterable[DcT],  # list[_DcT['*shape']]
    *,
    axis: int = 0,
) -> DcT:  # _DcT['len(arrays) *shape']:
  """Stack dataclasses together."""
  arrays = list(arrays)
  first_arr = arrays[0]

  if not isinstance(first_arr, array_dataclass.DataclassArray):
    raise TypeError('`v3d.stack` expect list of `v3d.DataclassArray`. Got '
                    f'{type(first_arr)}')

  # This might have some edge cases if user try to stack subclasses
  types = py_utils.groupby(
      arrays,
      key=type,
      value=lambda x: type(x).__name__,
  )
  if False in types:
    raise TypeError(
        f'v3.stack got conflicting types as input: {list(types.values())}')

  xnp = first_arr.xnp
  # If axis < 0, normalize the axis such as the last axis is before the inner
  # shape
  axis = np_utils.to_absolute_axis(axis, ndim=first_arr.ndim + 1)

  # Iterating over only the fields of the `first_arr` will skip optional fields
  # if those are not set in `first_arr`, even if they are present in others.
  # But is consistent with `jax.tree_map`:
  # jax.tree_map(lambda x, y: x+y, (None, 10), (1, 2)) == (None, 12)
  # Similarly, static values will be the ones from the first element.
  # pyformat: disable
  merged_arr = first_arr._map_field(  # pylint: disable=protected-access
      array_fn=
      lambda f: xnp.stack([getattr(arr, f.name) for arr in arrays], axis=axis),
      dc_fn=lambda f: stack([getattr(arr, f.name) for arr in arrays]),
  )
  # pyformat: enable
  return merged_arr

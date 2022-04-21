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

"""Python utils."""

from __future__ import annotations

import collections
import dataclasses
import importlib
import types
from typing import Any, Callable, Iterable, Optional, Type, TypeVar, Union

# If these are reused, we could move them to `epy`.

_T = TypeVar('_T')
_SelfOrCls = Union[Any, Type[Any]]

_K = TypeVar('_K')
_Tin = TypeVar('_Tin')
_Tout = TypeVar('_Tout')


def supports_protocol(
    self_or_cls: _SelfOrCls,
    protocol_fn_name: str,
) -> bool:
  """Returns `True` if the class support the protocol."""
  return (
      hasattr(self_or_cls, protocol_fn_name)
      and callable(getattr(self_or_cls, protocol_fn_name))
  )


# Is it possible to dynamically add TypeGuard+Protocol annotations ? (PEP 647)
def assert_supports_protocol(
    self_or_cls: _SelfOrCls,
    protocol_fn_name: str,
    *,
    msg: str = '',
) -> None:
  """Returns `True` if the class support the protocol."""
  if not supports_protocol(self_or_cls, protocol_fn_name):
    cls = _get_class(self_or_cls)
    raise NotImplementedError(
        f'`{cls.__qualname__}` does not implement the `.{protocol_fn_name}()` '
        f'protocol. {msg}')


def _get_class(self_or_cls: Union[_T, Type[_T]]) -> Type[_T]:
  if isinstance(self_or_cls, type):
    return self_or_cls
  else:
    return type(self_or_cls)


@dataclasses.dataclass
class LazyModule:
  """Module loaded lazily during first call."""
  module_name: str
  module: Optional[types.ModuleType] = None

  def __getattr__(self, name: str) -> Any:
    if self.module is None:  # Load on first call
      self.module = importlib.import_module(self.module_name)
    return getattr(self.module, name)


def identity(x: _Tin) -> _Tin:
  """Pass through function."""
  return x


def groupby(
    iterable: Iterable[_Tin],
    *,
    key: Callable[[_Tin], _K],
    value: Callable[[_Tin], _Tout] = identity,
) -> dict[_K, list[_Tout]]:
  """Similar to `itertools.groupby` but return result as a `dict()`.

  Example:

  ```python
  out = py_utils.groupby(
      ['555', '4', '11', '11', '333'],
      key=len,
      value=int,
  )
  # Order is consistent with above
  assert out == {
      3: [555, 333],
      1: [4],
      2: [11, 11],
  }
  ```

  Other difference with `itertools.groupby`:

   * Iterable do not need to be sorted. Order of the original iterator is
     preserved in the group.
   * Transformation can be applied to the value too

  Args:
    iterable: The iterable to group
    key: Mapping applied to group the values (should return a hashable)
    value: Mapping applied to the values

  Returns:
    The dict
  """
  groups = collections.defaultdict(list)
  for v in iterable:
    groups[key(v)].append(value(v))
  return dict(groups)


class _Ellipsis:
  """Ellipsis with repr as `...`. Used for better debug message."""

  def __repr__(self) -> str:
    return '...'


Ellipsis = _Ellipsis()  # pylint: disable=redefined-builtin

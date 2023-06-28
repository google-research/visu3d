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

"""Python utils."""

from __future__ import annotations

import dataclasses
import importlib
import types
from typing import Any, Optional, Type, TypeVar, Union

# If these are reused, we could move them to `epy`.

_T = TypeVar('_T')
_SelfOrCls = Union[Any, Type[Any]]


def supports_protocol(
    self_or_cls: _SelfOrCls,
    protocol_fn_name: str,
) -> bool:
  """Returns `True` if the class support the protocol."""
  return hasattr(self_or_cls, protocol_fn_name) and callable(
      getattr(self_or_cls, protocol_fn_name)
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
        f'protocol. {msg}'
    )


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

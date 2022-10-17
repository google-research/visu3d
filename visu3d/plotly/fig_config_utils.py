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

"""Global fig config utils."""

from __future__ import annotations

import dataclasses
from typing import Callable, Generic, Optional, TypeVar, Union

from etils import edc

_T = TypeVar('_T')


class LazyValue(Generic[_T]):
  """Lazyly compute a value."""

  def __init__(self, value_fn: Callable[[], _T]):
    self._value_fn = value_fn

  @property
  def value(self) -> _T:
    return self._value_fn()

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self.value})'


@edc.dataclass
@dataclasses.dataclass
class FigConfig:
  """Figure configuration options.

  Can be mutated globally, like:

  ```python
  v3d.fig_config.show_zero = False
  v3dfig_config..num_samples_point3d = None  # Do not subsample point display
  ```

  Attributes:
    show_zero: Whether to show the `(0, 0, 0)` origin, otherwise the plot x, y,
      z axis adapt to the data.
    num_samples_point3d: Max number of `v3d.Point3d` displayed by default (
      `None` for all)
    num_samples_point2d: Max number of `v3d.Point2d` displayed by default (
      `None` for all)
    num_samples_ray: Max number of `v3d.Ray` displayed by default (`None` for
      all)
  """

  show_zero: bool = True
  # TODO(epot): Currently this does not support `v3d.make_fig(num_samples_ray=)`
  num_samples_point3d: Optional[int] = 10_000
  num_samples_point2d: Optional[int] = 50_000
  num_samples_ray: Optional[int] = 500

  def replace(self: _T, **kwargs) -> _T:
    return dataclasses.replace(self, **kwargs)


fig_config = FigConfig()


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class TraceConfig:
  """Configuration of a single v3d object.

  Attributes:
    name: The name of the figure.
    num_samples: Maximum number of X to display (`-1` to display all). Keep
      rendering time reasonable by displaying only a subset of the total X.
  """

  # NOTE: When adding new properties here, please also update all
  # `.replace_fig_config(` function to get type checking/auto-complete.

  name: Optional[str] = None
  num_samples: Union[LazyValue[Optional[int]], Optional[int]] = -1

  def replace(self: _T, **kwargs) -> _T:
    return dataclasses.replace(self, **kwargs)

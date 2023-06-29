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

"""Global fig config utils."""

from __future__ import annotations

import dataclasses
import typing
from typing import Callable, Generic, Optional, TypeVar

from etils import edc

_T = TypeVar('_T')


@edc.dataclass
@dataclasses.dataclass
class FigConfig:
  """Figure configuration options.

  Can be mutated globally, like:

  ```python
  v3d.fig_config.show_zero = False
  v3d.fig_config.num_samples_point3d = None  # Do not subsample point display
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
    cam_scale: Scale of the cameras.
  """

  # When updating this, also update `v3d.make_fig` (for auto-complete /
  # discoverability)

  show_zero: bool = True
  num_samples_point3d: Optional[int] = 10_000
  num_samples_point2d: Optional[int] = 50_000
  num_samples_ray: Optional[int] = 500
  cam_scale: float = 1.0

  def replace(self: _T, **kwargs) -> _T:
    # Filter `...` (forwarded from `v3d.make_fig`)
    kwargs = {k: v for k, v in kwargs.items() if v is not ...}
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
  num_samples: Optional[int] = None

  # Hidden reference to the global `fig_config`.
  # This allow to locally overwrite the default `fig_config`, like:
  # `v3d.make_fig(rays, num_samples_ray=None)`
  # This is used for the lazy values
  _fig_config: FigConfig = dataclasses.field(
      default_factory=lambda: fig_config, repr=False
  )

  if not typing.TYPE_CHECKING:

    def __getattribute__(self, name: str):
      # Auto-resolve lazy values
      value = super().__getattribute__(name)
      if isinstance(value, LazyValue):
        return value.get_value(self._fig_config)
      else:
        return value

  def replace(self: _T, **kwargs) -> _T:
    """Alias for `dataclasses.replace`."""
    # Use a custom replace because `dataclasses.replace` remove the `LazyValue`

    def _get_attr(name: str) -> _T:
      if name in kwargs:
        return kwargs[name]
      else:
        return super(TraceConfig, self).__getattribute__(name)

    init_kwargs = {
        f.name: _get_attr(f.name) for f in dataclasses.fields(self) if f.init
    }
    return type(self)(**init_kwargs)


class LazyValue(Generic[_T]):
  """Lazyly compute a value."""

  def __init__(self, value_fn: Callable[[FigConfig], _T]):
    self._value_fn = value_fn

  def get_value(self, fig_config_: FigConfig) -> _T:
    return self._value_fn(fig_config_)

  def __repr__(self) -> str:
    # Technically, this should be the `object._fig_config` on which the
    # `LazyValue` is attached. Could be implemented with descriptor.
    return f'{type(self).__name__}({self._value_fn(fig_config)})'

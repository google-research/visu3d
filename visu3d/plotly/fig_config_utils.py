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
from typing import Optional, TypeVar

from etils import edc
from visu3d import dc_arrays

_T = TypeVar('_T')


@edc.dataclass
@dataclasses.dataclass
class _FigConfig:
  """Figure configuration options (base class)."""

  show_zero: bool = True
  # The values are directly fetched at the source, so set init == False
  # TODO(epot): Currently this does not support `v3d.make_fig(num_samples_ray=)`
  num_samples_point3d: int = dataclasses.field(init=False)
  num_samples_point2d: int = dataclasses.field(init=False)
  num_samples_ray: int = dataclasses.field(init=False)

  def replace(self: _T, **kwargs) -> _T:
    return dataclasses.replace(self, **kwargs)


class FigConfig(_FigConfig):
  """Figure configuration options.

  Can be mutated globally, like:

  ```python
  v3d.fig_config.show_zero = False
  ```

  Attributes:
    show_zero: Whether to show the `(0, 0, 0)` origin, otherwise the plot x, y,
      z axis adapt to the data.
    num_samples_point3d: Max number of v3d.Point3d displayed by default (-1 for
      all)
    num_samples_point2d: Max number of v3d.Point2d displayed by default (-1 for
      all)
    num_samples_ray: Max number of v3d.Ray displayed by default (-1 for all)
  """

  @property
  def num_samples_point3d(self) -> int:
    return dc_arrays.point.Point3d.fig_config.num_samples

  @property
  def num_samples_point2d(self) -> int:
    return dc_arrays.point.Point2d.fig_config.num_samples

  @property
  def num_samples_ray(self) -> int:
    return dc_arrays.ray.Ray.fig_config.num_samples

  # Use setattr to mutate the frozen dataclasses

  @num_samples_point3d.setter
  def num_samples_point3d(self, value: int):
    object.__setattr__(dc_arrays.point.Point3d.fig_config, 'num_samples', value)

  @num_samples_point2d.setter
  def num_samples_point2d(self, value: int):
    object.__setattr__(dc_arrays.point.Point2d.fig_config, 'num_samples', value)

  @num_samples_ray.setter
  def num_samples_ray(self, value: int):
    object.__setattr__(dc_arrays.ray.Ray.fig_config, 'num_samples', value)


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
  num_samples: int = -1

  def replace(self: _T, **kwargs) -> _T:
    return dataclasses.replace(self, **kwargs)

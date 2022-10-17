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

_T = TypeVar('_T')


@edc.dataclass
@dataclasses.dataclass
class FigConfig:
  """Figure configuration options.

  Can be mutated globally, like:

  ```python
  v3d.fig_config.show_zero = False
  ```

  Attributes:
    show_zero: Whether to show the `(0, 0, 0)` origin, otherwise the plot x, y,
      z axis adapt to the data.
  """

  show_zero: bool = True

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

  # TODO(epot): Could `num_samples` be made automatically applied for all
  # dataclass_array, in `plotly.make_traces` (might have performance issue
  # if many fields to subsample when only a few are actually used for display).
  # TODO(epot): More dynamic sub-sampling controled in `v3d.make_fig` per
  # figures (global control)

  name: Optional[str] = None
  num_samples: int = -1

  def replace(self: _T, **kwargs) -> _T:
    return dataclasses.replace(self, **kwargs)

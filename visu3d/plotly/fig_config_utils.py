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

from etils import edc


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

  def replace(self, **kwargs) -> FigConfig:
    return dataclasses.replace(self, **kwargs)


fig_config = FigConfig()

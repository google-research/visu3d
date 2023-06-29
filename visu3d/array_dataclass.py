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

"""Dataclass array wrapper."""

from __future__ import annotations

import dataclasses
from typing import Any

import dataclass_array as dca
from dataclass_array.typing import DcT
from visu3d.plotly import fig_config_utils
from visu3d.plotly import fig_utils


@dca.dataclass_array(broadcast=True, cast_dtype=True)
class DataclassArray(dca.DataclassArray, fig_utils.Visualizable):
  """Wrapper around `dca.DataclassArray` for all v3d objects.

  This class is like `dca.DataclassArray` but in addition:

  * Add the `my_obj.fig` property to all objects.
  * Add a `my_obj.fig_config` property to control object display options. As
    `dca.DataclassArray` are immutable, options can be updated using
    `my_obj = my_obj.replace_fig_config(**options)`.

  """

  __dca_non_init_fields__ = ('fig_config',)

  # Note: Because `FigConfig` is immutable, it is safe to use a shared instance
  # to avoid unecessary copy.
  fig_config: fig_config_utils.TraceConfig = dataclasses.field(
      default=fig_config_utils.TraceConfig(),
      repr=False,
      init=False,
  )

  def replace_fig_config(
      self: DcT,
      *,
      name: str = ...,  # pytype: disable=annotation-type-mismatch
      num_samples: int = ...,  # pytype: disable=annotation-type-mismatch
      **kwargs: Any,
  ) -> DcT:
    """Returns a copy of self with figure params overwritten."""
    fig_config_kwargs = dict(
        name=name,
        num_samples=num_samples,
        **kwargs,
    )
    # Filter Ellipsis values
    fig_config_kwargs = {
        k: v for k, v in fig_config_kwargs.items() if v is not ...
    }
    return self.replace(fig_config=self.fig_config.replace(**fig_config_kwargs))

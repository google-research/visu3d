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

"""Set plotly trace names."""

from __future__ import annotations

import collections
import dataclasses
import itertools

from etils import enp
from visu3d.plotly import fig_utils
from visu3d.utils.lazy_imports import plotly_base


@dataclasses.dataclass
class TraceNamer:
  """Set plotly trace names."""

  name_to_count: dict[str, itertools.count] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(itertools.count),
  )

  def set_name(
      self,
      traces: list[plotly_base.BaseTraceType],
      array: fig_utils.VisualizableInterface,
  ) -> None:
    """Update the name of the traces."""
    if not traces:  # No traces
      return

    elif fig_utils.is_visualizable(array):
      if fig_utils.has_fig_config(array) and array.fig_config.name:
        # User-defined name
        name = array.fig_config.name
      else:
        name = type(array).__name__
    elif enp.lazy.is_array(array):
      name = 'points'
    else:
      raise TypeError(f'Unexpected trace {type(array)}')

    curr_id = next(self.name_to_count[name])

    # Only update the name of the first trace
    # In the future, could have more complicated heuristics based on cases
    for trace in traces:
      id_suffix = f' {curr_id}' if curr_id else ''
      if not trace.name:
        trace.name = f'{name}{id_suffix}'
      # Set the group so all items are triggered together
      trace.legendgroup = f'{name}{id_suffix}'

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

"""Auto plot utils."""

from __future__ import annotations

from visu3d.plotly import fig_utils
from visu3d.utils.lazy_imports import IPython


def auto_plot_figs() -> None:
  """Auto-display `tuple[Visualizable, ...]` as figure.

  After this function is called, tuple of `v3d.Visualizable` objects are
  displayed directly as if they were passed to `v3d.make_fig`.

  >>> rays, cams, points

  Is an alias for

  >>> v3d.make_fig([rays, cams, points])

  It uses a simple heuristic (only check is the first element is visualizable).
  Other tuple are display as-is.

  """
  ipython = IPython.get_ipython()
  if ipython is None:
    return  # Non-notebook environement

  def make_fig(val):
    # Only check the first 5 elements
    if not val or not any(fig_utils.is_visualizable(v) for v in val[:5]):
      return None
    fig_utils.make_fig(list(val))._ipython_display_()  # pylint: disable=protected-access
    return ''

  print('Display `tuple[v3d.Visualizable, ...]` as figure')
  formatter = ipython.display_formatter.formatters['text/html']
  formatter.for_type(tuple, make_fig)

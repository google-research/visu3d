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

"""Plotly utils."""

from __future__ import annotations

import abc
from collections.abc import Sequence  # pylint: disable=g-importing-member
from typing import Any, Dict, List, Optional, Union

import dataclass_array as dca
import einops
from etils import enp
from etils.array_types import Array, FloatArray  # pylint: disable=g-multiple-import
import numpy as np
from visu3d import array_dataclass as v3d_dataclass_array
from visu3d import math
from visu3d.plotly import fig_config_utils
from visu3d.plotly import traces_builder
from visu3d.utils import py_utils
from visu3d.utils.lazy_imports import plotly_base
from visu3d.utils.lazy_imports import plotly_go as go

_Primitive = Union[str, int, bool, float]
_PlotlyKwargs = Dict[str, Union[np.ndarray, _Primitive]]

del abc  # TODO(epot): Why pytype doesn't like abc.ABC ?

_MARKER_SYMBOLS = {
    'circle',
    'circle-open',
    'cross',
    'diamond',
    'diamond-open',
    'square',
    'square-open',
    'x',
}

# TODO(epot): More dynamic sub-sampling for points ?
# * controled in `v3d.make_fig`
# * globally assigned (collect the global batch shape)
# * Add a tqdm bar ?

# TODO(epot): Refactor this file in 3 smaller files:
# * `traces_utils.py`: The `make_xyz` functions
# * `fig_utils.py`: The `make_fig` function
# * `visualization`: `Visualizable` & typing annotations

# Any class with the `make_traces` protocol
# TODO(epot): Could use protocol instead
VisualizableInterface = Any


class Visualizable:  # (abc.ABC):
  """Interface for elements which are visualizable."""

  # @abc.abstractmethod
  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    """Construct the traces of the given object."""
    raise NotImplementedError(
        f'To be visualizable, {type(self).__qualname__} should implement '
        'the `.make_traces()` protocol.'
    )

  @property
  def fig(self) -> go.Figure:
    """Construct the figure of the given object."""
    return make_fig([self])


VisualizableItem = Union[Visualizable, Array[...], VisualizableInterface]
VisualizableArg = Union[VisualizableItem, List[VisualizableItem]]


# TODO(epot): Potential improvement:
# * Accept a `dict[str, data]` (to have trace name)
# * Allow wrapping data in some `v3d.FigData(data, **kwargs)` to allow
#   customize metadata (color, point name,...) ?
# * Allow nested structure to auto-group multiple traces ?
def make_fig(
    *data: VisualizableArg,
    # Those arguments match `v3d.fig_config` options
    # They are duplicated for discoverability / auto-complete
    show_zero: bool = ...,  # pytype: disable=annotation-type-mismatch
    num_samples_point3d: Optional[int] = ...,  # pytype: disable=annotation-type-mismatch
    num_samples_point2d: Optional[int] = ...,  # pytype: disable=annotation-type-mismatch
    num_samples_ray: Optional[int] = ...,  # pytype: disable=annotation-type-mismatch
    cam_scale: float = ...,  # pytype: disable=annotation-type-mismatch
    **fig_config_kwargs: Any,
) -> go.Figure:
  """Returns the figure from the given data.

  ```python
  v3d.make_fig([obj0, obj1])  # Or `v3d.make_fig(obj0, obj1)`
  ```

  The figure will use `v3d.fig_config` global figure configuration. Individual
  config options can be overwritten here through kwargs.

  Args:
    *data: The data to plot. Either a `v3d.Vizualizable` or a `np.array` point
      cloud, or a list of the above.
    show_zero: Whether to show the `(0, 0, 0)` origin, otherwise the plot x, y,
      z axis adapt to the data.
    num_samples_point3d: Max number of `v3d.Point3d` displayed by default (
      `None` for all)
    num_samples_point2d: Max number of `v3d.Point2d` displayed by default (
      `None` for all)
    num_samples_ray: Max number of `v3d.Ray` displayed by default (`None` for
      all)
    cam_scale: Scale of the cameras.
    **fig_config_kwargs: Additional figure options (see `v3d.fig_config`)

  Returns:
    The plotly `go.Figure`, can be further modified.
  """
  traces = make_traces(
      *data,
      show_zero=show_zero,
      num_samples_point3d=num_samples_point3d,
      num_samples_point2d=num_samples_point2d,
      num_samples_ray=num_samples_ray,
      cam_scale=cam_scale,
      **fig_config_kwargs,
  )
  fig = go.Figure(data=traces)

  is_2d = _is_traces_2d(traces)

  if not _show_legend(traces):  # Eventually disable the legend
    fig.update_traces(showlegend=False)

  fig.update_layout(
      margin=dict(l=0, r=0, t=0, b=0),
      legend=dict(
          y=0.96,  # Move legend bellow to avoid overlapp with title
      ),
  )

  if is_2d:
    fig.update_yaxes(
        scaleanchor='x',
        scaleratio=1,
        autorange='reversed',
    )
  else:
    # TODO(epot): To avoid axes to flicker when toogle a trace, we should
    # fix the range, ratio,... of the axes to the max data boundaries.
    fig.update_scenes(
        aspectmode='data',  # Keep equal axis
        # TODO(plotly): Ideally, should set `xaxis_rangemode` rather than
        # having to add `make_zero_point`. But plotly is buggy.
        # Extend axis to include 0
        # xaxis_rangemode='tozero',
        # yaxis_rangemode='tozero',
        # zaxis_rangemode='tozero',
    )
  return fig


def make_traces(
    *data: VisualizableArg,
    **fig_config_kwargs: Any,
) -> list[plotly_base.BaseTraceType]:
  """Returns the traces from the given data."""
  #
  if len(data) == 1:  # `v3d.make_traces([a, b, c])` or `v3d.make_traces(a)`
    (data,) = data
    if not isinstance(data, (tuple, list)):
      data = [data]
  # Otherwise, called as `v3d.make_traces(a, b, c)`

  # TODO(epot): `curr_config` should be created in `make_fig` and
  # propagated downstream.
  curr_config = fig_config_utils.fig_config.replace(**fig_config_kwargs)

  # Mapping to count the number of trace of a type
  trace_namer = traces_builder.TraceNamer()

  # TODO(epot): Should dynamically sub-sample across all traces, instead of
  # subsampling individual traces.
  traces = []
  for val in data:
    if is_visualizable(val):
      if isinstance(val, dca.DataclassArray):
        if has_fig_config(val):
          # Overwrite the global fig_config with the local copy
          if isinstance(val, v3d_dataclass_array.DataclassArray):
            val = val.replace_fig_config(_fig_config=curr_config)
          val = math.subsample(
              val,
              num_samples=val.fig_config.num_samples,
              # TODO(epot): Should likely make the seed depend on other
              # factors like class name, position in make_traces *args,...
              seed=0,
          )
        val = val.as_np()
      sub_traces = val.make_traces()  # pytype: disable=attribute-error
      # Normalizing trace
      if isinstance(sub_traces, plotly_base.BaseTraceType):
        sub_traces = [sub_traces]
      trace_namer.set_name(sub_traces, val)
      traces.extend(sub_traces)
    elif enp.lazy.is_array(val) or isinstance(val, list):
      val = np.asarray(val)
      sub_traces = make_points(val, num_samples=curr_config.num_samples_point3d)
      trace_namer.set_name(sub_traces, val)
      traces.extend(sub_traces)
    elif isinstance(val, plotly_base.BaseTraceType):  # Already a trace
      traces.append(val)
    else:
      raise TypeError(f'Unsuported {type(val)}')
  if curr_config.show_zero and not _is_traces_2d(traces):
    traces.append(make_zero_point())
  return traces


def make_points(
    coords: FloatArray['*d num_dim'],
    *,
    color: Array['*d channel'] = None,
    num_samples: Optional[int] = None,
) -> list[plotly_base.BaseTraceType]:
  """Display a 2d or 3d point cloud.

  Args:
    coords: The point coordinates
    color: The optional point colors. Can be RGB uint8 value, or scalar float
      scale.
    num_samples: Returns the number of samples

  Returns:
    The plotly trace to display
  """
  if coords.shape[-1] not in (2, 3):
    raise ValueError(
        'Points should be `(..., 2)` or `(..., 3)`. '
        f'now. Got shape={coords.shape}'
    )
  if color is not None:
    assert color.shape[:-1] == coords.shape[:-1]
    color = color.reshape((-1, color.shape[-1]))
  coords = coords.reshape((-1, coords.shape[-1]))

  # TODO(epot): Subsample array if nb points >500
  coords, color = subsample(coords, color, num_samples=num_samples)  # pylint: disable=unbalanced-tuple-unpacking

  if color is not None:
    color = _normalize_color(color)

  if coords.shape[-1] == 3:
    point_cloud = _make_scatter_3d(coords, color=color)
  elif coords.shape[-1] == 2:
    point_cloud = _make_scatter_2d(coords, color=color)
  else:
    raise AssertionError
  return [point_cloud]


def _normalize_color(color: Array['*d channel']):
  """Normalize the color to plotly compatible arg."""
  if color.shape[-1] == 3:
    return [f'rgb({r}, {g}, {b})' for r, g, b in color]
  elif color.shape[-1] == 1:
    return color
  else:
    raise ValueError('Color should be (..., 1) or (..., 3)')


def _make_scatter_3d(
    coords: FloatArray['*d 3'],
    *,
    color: Array['*d channel'] = None,
) -> go.Scatter3d:
  """Returns 3d scatter plots."""
  points_xyz_kwargs = to_xyz_dict(coords)
  return go.Scatter3d(
      **points_xyz_kwargs,
      mode='markers',
      marker=go.scatter3d.Marker(
          size=2.0,
          color=color,
      ),
  )


def _make_scatter_2d(
    coords: FloatArray['*d 2'],
    *,
    color: Array['*d channel'] = None,
) -> go.Scattergl:
  """Returns 2d scatter plots."""
  # TODO(epot): Correct axis (x, y) to match numpy !!
  # Currently plotly use the `(y, x)` convention. I could not find a way to
  # invert
  points_xy_kwargs = to_xyz_dict(coords, names='xy')
  return go.Scattergl(
      **points_xy_kwargs,
      mode='markers',
      marker=go.scattergl.Marker(
          size=2.0,
          color=color,
      ),
  )


def make_lines_traces(
    start: FloatArray['*lines 3'],
    end: FloatArray['*lines 3'],
    *,
    axis: int = -1,
    end_marker: Optional[str] = None,
) -> list[plotly_base.BaseTraceType]:
  """Trace independent lines.

  Args:
    start: (x, y, z) coordinates of the start points
    end: (x, y, z) coordinates of the end points
    axis: Axis on which the (x, y, z) coordinates are defined
    end_marker: Marker at the end line. Can be any `go.Scatter3d.marker.symbol`
      value, or `'cone'`.

  Returns:
    The list of plotly traces
  """
  lines_xyz_kwargs = make_lines_kwargs(
      start=start,
      end=end,
      axis=axis,
      end_marker=end_marker,
  )
  lines_trace = go.Scatter3d(**lines_xyz_kwargs)
  traces = [lines_trace]
  if end_marker is None or end_marker in _MARKER_SYMBOLS:
    pass
  elif end_marker == 'cone':
    cone_kwargs = make_cones_kwargs(
        start=start,
        direction=end - start,
        axis=axis,
    )
    cone_traces = go.Cone(
        **cone_kwargs,
        showlegend=False,
        showscale=False,
        # TODO(plotly): Absolute size currently broken:
        # https://github.com/plotly/plotly.js/issues/3613
        sizemode='absolute',  # Not sure what's the difference with `scaled`
        sizeref=0.5,
        # TODO(epot): Add color
        # colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']]
    )
    traces.append(cone_traces)
  else:
    raise ValueError(f'Invalid end_marker={end_marker!r}')
  return traces


def make_lines_kwargs(
    start: FloatArray['... 3'],
    end: FloatArray['... 3'],
    *,
    axis: int = -1,
    end_marker: Optional[str] = None,
) -> _PlotlyKwargs:
  """Returns the kwargs to plot lines."""
  assert axis == -1
  # 1) Flatten the arrays
  # Shape is `*d 3`
  assert start.shape == end.shape
  assert start.shape[-1] == 3

  start = start.reshape((-1, 3))
  end = end.reshape((-1, 3))

  # 2) Build the lines
  lines_xyz = [[], [], []]
  for s, e in zip(start, end):
    for i in range(3):  # (x, y, z)
      lines_xyz[i].append(s[i])
      lines_xyz[i].append(e[i])
      lines_xyz[i].append(None)  # line break

  # Add the marker kwargs
  if end_marker in _MARKER_SYMBOLS:
    marker_kwargs = dict(
        mode='lines+markers',
        marker=go.scatter3d.Marker(
            size=[0, 6, 0] * len(start),
            symbol=end_marker,
        ),
    )
  else:
    marker_kwargs = dict(mode='lines')

  return dict(
      **to_xyz_dict(lines_xyz, axis=0),
      **marker_kwargs,
  )


def make_cones_kwargs(
    start: FloatArray['... 3'],
    direction: FloatArray['... 3'],
    *,
    start_ratio: float = 0.98,
    axis: int = -1,
) -> _PlotlyKwargs:
  """Returns the kwargs to plot cones."""
  assert axis == -1
  # 1) Flatten the arrays
  # Shape is `*d 3`
  assert start.shape == direction.shape
  assert start.shape[-1] == 3

  start = start.reshape((-1, 3))
  direction = direction.reshape((-1, 3))

  # 2) Build the lines
  xyz = start + start_ratio * direction
  uvw = direction
  return {
      **to_xyz_dict(xyz),
      **to_xyz_dict(uvw, names='uvw'),
  }


def make_zero_point() -> plotly_base.BaseTraceType:
  """Returns the trace corresponding to the origin (0, 0, 0) point."""
  return go.Scatter3d(
      x=[0],
      y=[0],
      z=[0],
      showlegend=False,  # Do not display the legend
      marker=go.scatter3d.Marker(
          size=0.01,  # Small point to avoid accidental selection
          # Transparent color (somehow opacity=0. does not work)
          color='rgba(0, 0, 0, 0.0)',
      ),
  )


def _show_legend(traces: List[plotly_base.BaseTraceType]) -> bool:
  """Returns `False` if only a single trace has a legend."""
  # Should also group legend per group ?
  # Default: `t.showlegend is None` -> show legend
  return len([t for t in traces if t.showlegend is not False]) > 1  # pylint: disable=g-bool-id-comparison


def to_xyz_dict(
    arr: Array['... 3'],
    *,
    pattern: str = '{}',
    names: Union[str, Sequence[str]] = 'xyz',
    axis: int = -1,
) -> _PlotlyKwargs:
  """Convert np.array to xyz dict.

  Useful to create plotly kwargs from numpy arrays.

  Example:

  ```python
  to_xyz_dict(np.zeros((1, 3))) == {'x': [0], 'y': [0], 'z': [0]}
  to_xyz_dict(
    [0, 1, 2],
    pattern='axis_{}'
    names='uvw',
  ) == {'axis_u': 0, 'axis_v': 1, 'axis_w': 2}
  ```

  Args:
    arr: Array to convert
    pattern: Pattern to use for the axis names
    names: Names of the axis (default to 'x', 'y', 'z')
    axis: Axis containing the x, y, z coordinates to dispatch

  Returns:
    xyz_dict: The dict containing plotly kwargs.
  """
  arr = np.asarray(arr)
  if arr.shape[axis] != len(names):
    raise ValueError(f'Invalid shape: {arr.shape}[{axis}] != {len(names)}')

  # Build the `dict(x=arr[..., 0], y=arr[..., 1], z=arr[..., 2])`
  vals = {
      pattern.format(axis_name): arr_slice.flatten()
      for axis_name, arr_slice in zip(names, np.moveaxis(arr, axis, 0))
  }
  # Normalize scalars (as plotly reject `np.array(1)`)
  vals = {k: v if v.shape else v.item() for k, v in vals.items()}
  return vals


def subsample(
    *arrays: Optional[Array['... d']],
    num_samples: Optional[int],
) -> list[Optional[Array['...']]]:
  """Flatten and subsample the arrays (keeping the last dimension)."""
  if num_samples is None or num_samples == -1:  # No sampling
    return list(arrays)
  assert arrays[0] is not None
  shape = arrays[0].shape
  assert len(shape) >= 1
  batch_size = dca.utils.np_utils.size_of(shape[:-1])

  if batch_size > num_samples:
    # All arrays are sub-sampled the same way, so generate ids separately
    rng = np.random.default_rng(0)
    idx = rng.choice(batch_size, size=num_samples, replace=False)

  arrays_out = []
  for arr in arrays:
    if arr is None:
      arrays_out.append(None)
      continue
    if arr.shape[:-1] != shape[:-1]:
      raise ValueError('Incompatible shape')
    arr = einops.rearrange(arr, '... d -> (...) d')  # Flatten
    if batch_size > num_samples:
      arr = arr[idx]
    arrays_out.append(arr)

  return arrays_out


# Might want to extand this to PlotType.PLOT_2D, PlotType.PLOT_3D,... ?
def _is_traces_2d(traces: list[plotly_base.BaseTraceType]) -> bool:
  """Returns `True` if the traces are a 2d plot."""
  traces_2d_cls = (  # TODO(epot): Complete all the plotly types
      go.Scatter,
      go.Scattergl,
  )
  has_2d = False
  has_3d = False
  for trace in traces:
    if isinstance(trace, traces_2d_cls):
      has_2d = True
    else:
      has_3d = True
  if has_2d and has_3d:
    cls_names = {type(trace).__name__ for trace in traces}
    raise ValueError(
        f'Trying to mix 2d and 3d plots: {cls_names}. Please open a bug if '
        'this is an issue.'
    )
  return has_2d


def is_visualizable(item: VisualizableItem) -> bool:
  """Returns `True` if the element is a visualizable item."""
  return py_utils.supports_protocol(item, 'make_traces')


def has_fig_config(item: VisualizableItem) -> bool:
  return isinstance(
      getattr(item, 'fig_config', None), fig_config_utils.TraceConfig
  )

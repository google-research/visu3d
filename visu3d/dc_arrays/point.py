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

"""Point3d utils."""

from __future__ import annotations

import dataclasses
import typing
from typing import Optional, Union

import dataclass_array as dca
from etils.array_types import FloatArray, ui8  # pylint: disable=g-multiple-import
from visu3d import array_dataclass
from visu3d import plotly
from visu3d.dc_arrays import transformation
from visu3d.plotly import fig_config_utils
from visu3d.utils import np_utils
from visu3d.utils.lazy_imports import plotly_base

if typing.TYPE_CHECKING:
  from visu3d.dc_arrays import camera_spec as camera_spec_lib


class Point3d(array_dataclass.DataclassArray):
  """3d point cloud.

  Attributes:
    p: 3d (x, y, z) coordinates
    rgb: uint8 color
  """

  p: FloatArray['*shape 3']
  rgb: Optional[ui8['*shape 3']] = None

  # Overwrite `v3d.DataclassArray.fig_config`.
  fig_config: plotly.TraceConfig = dataclasses.field(
      default=plotly.TraceConfig(
          num_samples=fig_config_utils.LazyValue(
              lambda fig_config: fig_config.num_samples_point3d
          )
      ),
      repr=False,
      init=False,
  )

  def __add__(self, translation: FloatArray['... 3']) -> Point3d:
    """Translate the position."""
    translation = self.xnp.asarray(translation)
    return self.replace(p=self.p + translation)

  __sub__ = np_utils.__sub__

  def clip(
      self,
      min: Optional[Union[FloatArray['3'], float]] = None,  # pylint: disable=redefined-builtin
      max: Optional[Union[FloatArray['3'], float]] = None,  # pylint: disable=redefined-builtin
  ) -> Point3d:
    """Clip the position coordinates to the (min, max) boundaries."""
    min = dca.utils.np_utils.asarray(min, xnp=self.xnp, optional=True)
    max = dca.utils.np_utils.asarray(max, xnp=self.xnp, optional=True)
    return self.replace(p=self.p.clip(min, max))

  # Protocols (inherited)

  def apply_transform(self, tr: transformation.Transform) -> Point3d:
    # No `color` modification
    return self.replace(p=tr.apply_to_pos(self.p))

  def apply_px_from_cam(self, spec: camera_spec_lib.CameraSpec) -> Point2d:
    """Apply the `px_from_cam @ self` transformation."""
    # TODO(epot): Expose with_depth in the public API
    p, depth = spec._px_and_depth_from_cam(self.p)  # pylint: disable=protected-access
    return Point2d(
        p=p,
        rgb=self.rgb,
        depth=depth,
    )

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    return plotly.make_points(self.p, color=self.rgb)


class Point2d(array_dataclass.DataclassArray):
  """2d point cloud.

  Attributes:
    p: 2d (x, y) == (u, v) == (w, h) coordinates
    rgb: uint8 color
    depth: The depth in camera coordinates.
  """

  p: FloatArray['*shape 2']
  depth: Optional[FloatArray['*shape 1']] = None
  rgb: Optional[ui8['*shape 3']] = None

  # Overwrite `v3d.DataclassArray.fig_config`.
  fig_config: plotly.TraceConfig = dataclasses.field(
      default=plotly.TraceConfig(
          num_samples=fig_config_utils.LazyValue(
              lambda fig_config: fig_config.num_samples_point2d
          )
      ),
      repr=False,
      init=False,
  )

  def clip(
      self,
      min: Optional[Union[FloatArray['2'], float]] = None,  # pylint: disable=redefined-builtin
      max: Optional[Union[FloatArray['2'], float]] = None,  # pylint: disable=redefined-builtin
  ) -> Point3d:
    """Clip the position coordinates to the (min, max) boundaries."""
    min = dca.utils.np_utils.asarray(min, xnp=self.xnp, optional=True)
    max = dca.utils.np_utils.asarray(max, xnp=self.xnp, optional=True)
    return self.replace(p=self.p.clip(min, max))

  # Protocols

  def apply_cam_from_px(
      self,
      spec: camera_spec_lib.CameraSpec,
  ) -> Point3d:
    """Apply the `px_from_cam @ self` transformation."""
    p = spec.cam_from_px @ self.p
    if self.depth is not None:
      p = p * self.depth
    return Point3d(
        p=p,
        rgb=self.rgb,
    )

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    traces = []

    (trace,) = plotly.make_points(self.p, color=self.rgb)
    traces.append(trace)

    if self.depth is not None:
      pass  # TODO(epot): Add depth trace (no displayed by default)
      # trace, = plotly.make_points(
      #     self.p,
      #     color=self.depth
      # )
      # traces.append(trace)

    return traces

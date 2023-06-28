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

"""Camera util."""

from __future__ import annotations

import dataclasses
from typing import Any

import dataclass_array as dca
from dataclass_array.typing import DcT
from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
import numpy as np
from visu3d import array_dataclass
from visu3d import plotly
from visu3d.dc_arrays import camera_spec
from visu3d.dc_arrays import point as point_lib
from visu3d.dc_arrays import ray as ray_lib
from visu3d.dc_arrays import transformation
from visu3d.utils import np_utils
from visu3d.utils.lazy_imports import plotly_base


class Camera(array_dataclass.DataclassArray):
  """A camera located in space.

  Attributes:
    spec: Camera intrinsics parameters
    world_from_cam: Camera pose (`v3d.Transformation`)
  """

  spec: camera_spec.CameraSpec
  world_from_cam: transformation.Transform

  # Overwrite `v3d.DataclassArray.fig_config`.
  fig_config: camera_spec.TraceConfig = dataclasses.field(
      default=camera_spec.TraceConfig(),
      repr=False,
      init=False,
  )

  @classmethod
  def from_look_at(
      cls,
      *,
      pos: FloatArray['*shape 3'],
      target: FloatArray['*shape 3'],
      spec: camera_spec.CameraSpec,
  ) -> Camera:
    """Factory which creates a camera looking at `target`.

    This assume the camera is parallel to the floor. See `v3d.CameraSpec`
    for axis conventions.

    Args:
      pos: Origin position
      target: Target position
      spec: Camera specifications.

    Returns:
      cam: Camera pointing to the ray.
    """
    world_from_cam = transformation.Transform.from_look_at(
        target=target,
        pos=pos,
    )
    return cls(spec=spec, world_from_cam=world_from_cam)

  @dca.vectorize_method
  def look_at(self, target: FloatArray['*shape 3']) -> Camera:
    """Returns a new camera looking at the target."""
    world_from_cam = self.world_from_cam.look_at(target)
    return self.replace(world_from_cam=world_from_cam)

  @property
  def resolution(self) -> tuple[int, int]:
    """`(h, w)` resolution in pixel."""
    return self.spec.resolution

  @property
  def hw(self) -> tuple[int, int]:
    """`(Height, Width)` in pixel (for usage in `(i, j)` coordinates)."""
    return self.spec.hw

  @property
  def wh(self) -> tuple[int, int]:
    """`(Width, Height)` in pixel (for usage in `(u, v)` coordinates)."""
    return self.spec.wh

  @property
  def h(self) -> int:
    """Height in pixel."""
    return self.spec.h

  @property
  def w(self) -> int:
    """Width in pixel."""
    return self.spec.w

  def __add__(self, translation: FloatArray['... 3']) -> Camera:
    """Translate the position."""
    return self.replace(world_from_cam=self.world_from_cam + translation)

  __sub__ = np_utils.__sub__

  @dca.vectorize_method(static_args={'normalize'})
  def rays(self, normalize: bool = True) -> ray_lib.Ray:
    """Creates the rays.

    Args:
      normalize: If `False`, returns camera rays in the `z=1` from the camera
        frame.

    Returns:
      rays: Pose
    """
    cam_dir = self.spec.cam_from_px @ self.spec.px_centers()
    # Rotate the points from cam -> world
    world_dir = self.world_from_cam.apply_to_dir(cam_dir)
    # Position is (0, 0, 0) so no need to transform

    rays = ray_lib.Ray(pos=self.world_from_cam.t, dir=world_dir)
    if normalize:
      rays = rays.normalize()
    return rays

  @property
  def cam_from_world(self) -> transformation.Transform:
    """Transformation from 3d world coordinates to 3d camera coordinates."""
    return self.world_from_cam.inv

  @property
  def px_from_world(self) -> transformation.TransformBase:
    """Transfomration from 3d world coordinates to 2d pixel coordinates."""
    return self.spec.px_from_cam @ self.cam_from_world

  @property
  def world_from_px(self) -> transformation.TransformBase:
    """Transfomration from 2d pixel coordinates to 3d world coordinates."""
    return self.world_from_cam @ self.spec.cam_from_px

  @dca.vectorize_method
  def render(self, points: point_lib.Point3d) -> FloatArray['*shape h w 3']:
    """Project 3d points to the camera screen.

    Args:
      points: 3d points.

    Returns:
      img: The projected 3d points.
    """
    # TODO(epot): Support float colors and make this differentiable!
    if not isinstance(points, point_lib.Point3d):
      raise TypeError(
          f'Camera.render expect `v3d.Point3d` as input. Got: {points}.'
      )

    # Project 3d -> 2d coordinates
    points2d = self.px_from_world @ points

    # Flatten pixels
    points2d = points2d.flatten()
    px_coords = points2d.p
    rgb = points2d.rgb

    # Compute the valid coordinates
    w_coords = px_coords[..., 0]
    h_coords = px_coords[..., 1]
    valid_coords_mask = (
        (0 <= h_coords)
        & (h_coords < self.h - 1)
        & (0 <= w_coords)
        & (w_coords < self.w - 1)
        & (points2d.depth[..., 0] > 0)  # Filter points behind the camera
    )
    rgb = rgb[valid_coords_mask]
    px_coords = px_coords[valid_coords_mask]
    px_coords = enp.compat.astype(enp.compat.round(px_coords), self.xnp.int32)

    # TODO(epot): Should we create a `xnp.asarray` ?
    # TODO(epot): The dtype should be cloned from point.rgb !
    img = np.zeros((*self.resolution, 3), dtype=np.uint8)
    # px_coords is (h, w)
    img[px_coords[..., 1], px_coords[..., 0]] = rgb
    return img

  # Could be removed but only kept for type-checking / auto-complete.
  def replace_fig_config(  # pylint: disable=useless-parent-delegation
      self: DcT,
      *,
      name: str = ...,  # pytype: disable=annotation-type-mismatch
      scale: float = ...,  # pytype: disable=annotation-type-mismatch
      **kwargs: Any,
  ) -> DcT:
    """Returns a copy of self with figure params overwritten."""
    return super().replace_fig_config(name=name, scale=scale, **kwargs)

  # Protocols (inherited)

  def apply_transform(self, tr: transformation.Transform) -> Camera:
    return self.replace(world_from_cam=tr @ self.world_from_cam)

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    spec = self.spec.replace(fig_config=self.fig_config)
    # TODO(epot): Add arrow to indicates the orientation ?
    start, end = spec._get_camera_lines()  # pylint: disable=protected-access
    start = self.world_from_cam @ start
    end = self.world_from_cam @ end
    return plotly.make_lines_traces(start=start, end=end)

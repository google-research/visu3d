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

"""Camera spec utils."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any, Optional, Tuple

import dataclass_array as dca
from dataclass_array.typing import DcOrArray, DcT  # pylint: disable=g-multiple-import
from etils import edc
from etils import enp
from etils import epy
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from visu3d import array_dataclass
from visu3d import math
from visu3d import plotly
from visu3d.dc_arrays import transformation
from visu3d.plotly import fig_config_utils
from visu3d.utils import np_utils
from visu3d.utils import py_utils
from visu3d.utils.lazy_imports import plotly_base

del abc  # TODO(epot): Why pytype don't like abc ?

assert_supports_protocol = functools.partial(
    py_utils.assert_supports_protocol,
    msg=(
        'Required to support camera transformations. See `v3d.Point3d` and '
        '`v3d.Point2d` for example.'
    ),
)


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class TraceConfig(plotly.TraceConfig):
  """Figure configuration.

  Attributes:
    scale: The scale of the camera.
  """

  scale: float = fig_config_utils.LazyValue(  # pytype: disable=annotation-type-mismatch
      lambda fig_config: fig_config.cam_scale
  )


class CameraSpec(array_dataclass.DataclassArray):  # (abc.ABC):
  """Camera intrinsics specification.

  Define the interface of camera model. See `PinholeCamera` for an example of
  class implementation.

  Support batching to allow to stack multiple cameras with the same resolution
  in a single `CameraSpec`.

  ```python
  specs = dca.stack([PinholeCamera(...) for _ in range(10)])

  isinstance(specs, CameraSpec)
  assert specs.shape == (10,)
  assert specs.px_centers().shape == (10, h, w, 2)
  ```

  CameraSpec also allow to project from/to pixel coordinates.

  ```python
  px_coords = specs.px_from_cam @ cam_coords
  ```

  This works with:

  * `xnp.asarray`: `(..., 3) -> (..., 2)`
  * `v3d.Point3d` -> `v3d.Point2d`
  * Your custom objects. To support this transformation, your dataclass should
    implement the protocols:

    * `apply_px_from_cam(self, camera_spec: v3d.CameraSpec)`:
      to support: `spec.px_from_cam @ my_obj`
    * `apply_cam_from_px(self, camera_spec: v3d.CameraSpec)`:
      to support: `spec.cam_from_px @ my_obj`

  Attributes:
    resolution: Camera resolution (in px).
    h: Camera height resolution (in px).
    w: Camera width resolution (in px).
    fig_config: Additional figure configuration.
  """

  resolution: Tuple[int, int]

  # Overwrite `v3d.DataclassArray.fig_config`.
  fig_config: TraceConfig = dataclasses.field(
      default=TraceConfig(),
      repr=False,
      init=False,
  )

  @property
  def h(self) -> int:
    return self.resolution[0]

  @property
  def w(self) -> int:
    return self.resolution[1]

  @property
  def hw(self) -> tuple[int, int]:
    """`(Height, Width)` in pixel (for usage in `(i, j)` coordinates)."""
    return (self.h, self.w)

  @property
  def wh(self) -> tuple[int, int]:
    """`(Width, Height)` in pixel (for usage in `(u, v)` coordinates)."""
    return (self.w, self.h)

  @property
  @transformation.custom_transform
  def px_from_cam(  # pylint: disable=property-with-parameters
      self,
      points3d: DcOrArray,
  ) -> transformation.TransformBase:  # pylint: disable=g-doc-args
    """Project camera 3d coordinates to px 2d coordinates.

    Usage:

    ```python
    pts2d = spec.px_from_cam @ pts3d
    ```

    Input can have arbitrary batch shape, including no batch shape for a
    single point as input.

    Returns:
      The transformation 3d cam coordinates -> 2d pixel coordinates.
    """
    points3d = dca.utils.np_utils.asarray(points3d, xnp=self.xnp)
    if isinstance(points3d, dca.DataclassArray):
      assert_supports_protocol(points3d, 'apply_px_from_cam')
      return points3d.apply_px_from_cam(self)
    else:
      return self._px_and_depth_from_cam(points3d)[0]

  # @abc.abstractmethod
  def _px_and_depth_from_cam(
      self,
      points3d,
  ) -> tuple[FloatArray['*d 2'], FloatArray['*d 1']]:
    """Project camera 3d coordinates to px 2d coordinates (internal).

    Args:
      points3d: 3d points

    Returns:
      `(point2d, depth)`
    """
    raise NotImplementedError

  @property
  @transformation.custom_transform
  def cam_from_px(  # pylint: disable=property-with-parameters
      self,
      points2d: FloatArray['*d 2'],
  ) -> transformation.TransformBase:  # pylint: disable=g-doc-args
    """Unproject 2d pixel coordinates in image space to camera space.

    Usage:

    ```python
    pts3d = spec.cam_from_px @ pts2d
    ```

    Note: Points returned by this function are not normalized. Points
    are returned at z=1 for pinhole camera.

    Input can have arbitrary batch shape, including no batch shape for a
    single point as input.

    Returns:
      The transformation 2d pixel coordinates -> 3d cam coordinates.
    """
    points2d = dca.utils.np_utils.asarray(points2d, xnp=self.xnp)
    if isinstance(points2d, dca.DataclassArray):
      assert_supports_protocol(points2d, 'apply_cam_from_px')
      return points2d.apply_cam_from_px(self)
    else:
      return self._cam_from_px(points2d)

  # @abc.abstractmethod
  def _cam_from_px(
      self,
      points2d: FloatArray['*d 2'],
  ) -> FloatArray['*d 3']:
    """Implemementation of `spec.cam_from_px @ ptss`."""
    raise NotImplementedError

  @dca.vectorize_method
  def px_centers(self) -> FloatArray['*shape h w 2']:
    """Returns 2D coordinates of centers of all pixels in the camera image.

    This camera model uses the conventions:

     * Top-left corner of the image is `(0, 0)`
     * Bottom-right corner is `(w, h)` (NOT `(h, w)`)
     * Pixels are centered, so `px_centers()[0, 0] == 0.5`

    Returns:
      2D image coordinates of center of all pixels of shape `(h, w, 2)`.
    """
    xnp = self.xnp
    coord_w, coord_h = xnp.meshgrid(
        xnp.arange(self.w),  # w
        xnp.arange(self.h),  # h
        indexing='xy',
    )
    points2d = xnp.stack([coord_w, coord_h], axis=-1)
    points2d = xnp.asarray(points2d, dtype=xnp.float32) + 0.5
    assert points2d.shape == (self.h, self.w, 2)
    return points2d

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

  # Protocols & internals

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    # TODO(epot): Add arrow to indicates the orientation ?
    start, end = self._get_camera_lines()
    return plotly.make_lines_traces(start=start, end=end)

  @dca.vectorize_method
  def _get_camera_lines(self) -> FloatArray['*shape 4 3']:
    corners_px = [  # Screen corners, in (u, v) coordinates
        [0, 0],
        [0, self.h],
        [self.w, 0],
        [self.w, self.h],
    ]
    corners_world = self.cam_from_px @ self.xnp.asarray(corners_px)
    corners_world = corners_world * self.fig_config.scale

    start = [
        # 4 lines from center -> corners
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        # 4 lines for the frame
        corners_world[0],
        corners_world[1],
        corners_world[2],
        corners_world[3],
    ]
    end = [
        # 4 lines from center -> corners
        corners_world[0],
        corners_world[1],
        corners_world[2],
        corners_world[3],
        # 4 lines for the frame
        corners_world[1],
        corners_world[3],
        corners_world[0],
        corners_world[2],
    ]
    return self.xnp.asarray(start), self.xnp.asarray(end)


class PinholeCamera(CameraSpec):
  """Simple camera model.

  Camera conventions:
  In camera/pixel coordinates:

  * u == x == j == w (orientation: →)
  * v == y == i == h (orientation: ↓)

  In camera frame coordinates:

  * `(0, 0, 1)` is at the center of the image
  * x: →
  * y: ↓
  * z is pointing forward

  Attributes:
    K: Camera intrinsics parameters.
    resolution: (h, w) resolution
  """

  K: FloatArray['*shape 3 3']  # pylint: disable=invalid-name

  @classmethod
  def from_focal(
      cls,
      *,
      resolution: tuple[int, int],
      focal_in_px: float,
      xnp: Optional[enp.NpModule] = None,
  ) -> PinholeCamera:
    """Camera factory.

    Args:
      resolution: `(h, w)` resolution in pixel
      focal_in_px: Focal length in pixel
      xnp: `numpy`, `jax.numpy` or `tf.experimental.numpy`. Numpy module to use.
        Default to `numpy`.

    Returns:
      A `PinholeCamera` instance with provided intrinsics.
    """
    if xnp is None:
      xnp = enp.lazy.get_xnp(focal_in_px, strict=False)

    # TODO(epot): Could provide more customizability
    # * Support `focal_in_mm` / `sensor_width`
    # * Support custom central point (cx, cy)
    # * Support different focal for w, h (fx, fy)
    # Warning: Which API when convensions are inconsistents ?
    # resolution == (h, w) but (fx, fy) == (fw, fh)

    # Central point in pixel (offset of the (0, 0) pixel)
    # Because our pixel coordinates are (0, 1), we set the central point
    # to the middle.
    h, w = resolution
    ch = h / 2  # h == y
    cw = w / 2  # w == x

    K = xnp.asarray(  # pylint: disable=invalid-name
        [
            [focal_in_px, 0, cw],  # cx
            [0, focal_in_px, ch],  # cy
            [0, 0, 1],
        ],
        dtype=xnp.float32,
    )
    return cls(
        K=K,
        resolution=resolution,
    )

  def _px_and_depth_from_cam(
      self,
      points3d,
  ):
    if points3d.shape[-1] != 3:
      raise ValueError(f'Expected cam coords {points3d.shape} to be (..., 3)')

    # K @ [X,Y,Z] -> s * [u, v, 1]
    # (3, 3) @ (..., 3) -> (..., 3)
    points2d = self.xnp.einsum('ij,...j->...i', self.K, points3d)
    # Normalize: s * [u, v, 1] -> [u, v, 1]
    # And only keep [u, v]
    depth = points2d[..., 2:3]
    points2d = points2d[..., :2] / (depth + 1e-8)
    return points2d, depth

  def _cam_from_px(
      self,
      points2d: FloatArray['*d 2'],
  ) -> FloatArray['*d 3']:
    assert not self.shape  # Should be vectorized
    points2d = dca.utils.np_utils.asarray(points2d, xnp=self.xnp)
    if points2d.shape[-1] != 2:
      raise ValueError(f'Expected pixel coords {points2d.shape} to be (..., 2)')

    # [u, v] -> [u, v, 1]
    # Concatenate (..., 2) with (..., 1) -> (..., 3)
    points2d = np_utils.append_row(points2d, 1.0, axis=-1)

    # [X,Y,Z] / s = K-1 @ [u, v, 1]
    # (3, 3) @ (..., 3) -> (..., 3)
    k_inv = enp.compat.inv(self.K)
    points3d = self.xnp.einsum('ij,...j->...i', k_inv, points2d)

    # TODO(epot): Option to return normalized rays ?
    # Set z to -1
    # [X,Y,Z] -> [X, Y, Z=1]
    points3d = points3d / enp.compat.expand_dims(points3d[..., 2], axis=-1)
    return points3d


class Spec360(CameraSpec):
  """Camera spec representing 360 panorama.

  Uses equirectangular projection for the 2d.
  """

  def _px_and_depth_from_cam(
      self,
      points3d,
  ):
    points3d = self._cv_from_wolfram.inv @ points3d
    depth, theta, phi = math.carthesian_to_spherical(points3d)

    # Scale angles to (0, h), (0, w)
    points2d = self.xnp.stack([theta, phi], axis=-1)
    points2d = points2d * self.xnp.asarray(
        [self.w / enp.tau, (2 * self.h) / enp.tau]
    )
    return points2d, depth[..., None]

  def _cam_from_px(
      self,
      points2d: FloatArray['*d 2'],
  ) -> FloatArray['*d 3']:
    # Normalize points to:
    # * i: [0, w] -> [0, tau]
    # * j: [0, h] -> [0, tau / 2]
    # Reminder, 2d pixels are in (u, v) coordinates (so `(w, h)`)
    # (h, w, 2)
    angles = points2d * self.xnp.asarray(
        [enp.tau / self.w, enp.tau / (2 * self.h)]
    )
    theta = angles[..., 0]
    phi = angles[..., 1]

    # Compute (x, y, z) direction
    points3d = math.spherical_to_carthesian(theta=theta, phi=phi)
    points3d = self._cv_from_wolfram @ points3d
    return points3d

  @epy.backports.cached_property
  def _cv_from_wolfram(self) -> transformation.Transform:
    """Convert Wolfram cathesian coordinates to OpenCv camera convention."""
    return transformation.Transform(
        R=self.xnp.asarray([
            [0, -1, 0],
            [0, 0, -1],
            [-1, 0, 0],
        ]),
    )

  # TODO(epot): Better camera plotly display!

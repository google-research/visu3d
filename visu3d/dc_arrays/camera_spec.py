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

"""Camera spec utils."""

from __future__ import annotations

import abc
import dataclasses
import functools
import typing
from typing import Optional, Tuple

import dataclass_array as dca
from dataclass_array.typing import DcOrArray
from etils import edc
from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from visu3d import array_dataclass
from visu3d import plotly
from visu3d.dc_arrays import transformation
from visu3d.utils import np_utils
from visu3d.utils import py_utils
from visu3d.utils.lazy_imports import plotly_base

del abc  # TODO(epot): Why pytype don't like abc ?

assert_supports_protocol = functools.partial(
    py_utils.assert_supports_protocol,
    msg='Required to support camera transformations. See `v3d.Point3d` and '
    '`v3d.Point2d` for example.',
)


@edc.dataclass(kw_only=True)
@dataclasses.dataclass(frozen=True)
class FigConfig:
  """Figure configuration.

  Attributes:
    scale: The scale of the camera.
  """
  scale: float = 1.0

  def replace(self, **kwargs) -> FigConfig:
    return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True)
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

  * `xnp.array`: `(..., 3) -> (..., 2)`
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
  __dca_non_init_fields__ = ('fig_config',)

  resolution: Tuple[int, int]
  # Note: Because `FigConfig` is immutable, it is safe to use a shared instance
  # to avoid unecessary copy.
  fig_config: FigConfig = dataclasses.field(
      default=FigConfig(),
      repr=False,
      init=False,
  )
  focal_length: Tuple[float, float]

  @property
  def h(self) -> int:
    return self.resolution[0]

  @property
  def w(self) -> int:
    return self.resolution[1]

  @property
  def hw(self) -> tuple[int, int]:
    """`(Height, Width)` in pixel (for usage in `(u, v)` coordinates)."""
    return (self.h, self.w)

  @property
  def wh(self) -> tuple[int, int]:
    """`(Width, Height)` in pixel (for usage in `(i, j)` coordinates)."""
    return (self.w, self.h)

  @property
  def fw(self) -> float:
    """Focal length (along x-axis) in pixels (for usage in `(i, j)` coordinates)."""
    return self.focal_length[0]

  @property
  def fh(self) -> float:
    """Focal length (along y-axis) in pixels (for usage in `(i, j)` coordinates)."""
    return self.focal_length[1]

  @property
  def focal_px_wh(self) -> tuple[float, float]:
    """Focal length in pixel (`(fw, fh)`)."""
    return (self.fw, self.fh)

  @property
  def focal_px(self) -> float:
    """Unique Focal length in pixels (when fw == fh)."""

    def _err_msg():
      return (
          'Cannot get `CameraSpec.focal_px` when fw and fh are '
          f'different: {self.focal_px_wh}'
      )

    if self.fw != self.fh:
      raise ValueError(_err_msg())

    return self.fw

  @property
  def fov_w(self) -> float:
    """Field of view (horizontal) in radians (for usage in `(i, j)` coordinates)."""
    return 2 * self.xnp.arctan(self.w / (2 * self.fw))

  @property
  def fov_h(self) -> float:
    """Field of view (vertical) in radians (for usage in `(i, j)` coordinates)."""
    return 2 * self.xnp.arctan(self.h / (2 * self.fh))

  @property
  def fov(self) -> float:
    """Field of view in radians (`(fov_w, fov_h)`)."""

    return (self.fov_w, self.fov_h)

  # @abc.abstractmethod
  @property
  def px_from_cam(self) -> transformation.TransformBase:
    """Project camera 3d coordinates to px 2d coordinates.

    Input can have arbitrary batch shape, including no batch shape for a
    single point as input.

    Returns:
      The transformation 3d cam coordinates -> 2d pixel coordinates.
    """
    raise NotImplementedError

  # @abc.abstractmethod
  @property
  def cam_from_px(self) -> transformation.TransformBase:
    """Unproject 2d pixel coordinates in image space to camera space.

    Note: Points returned by this function are not normalized. Points
    are returned at z=1.

    Input can have arbitrary batch shape, including no batch shape for a
    single point as input.

    Returns:
      The transformation 2d pixel coordinates -> 3d cam coordinates.
    """
    raise NotImplementedError

  # @abc.abstractmethod
  def px_centers(self) -> FloatArray['*shape h w 2']:
    """Returns 2D coordinates of centers of all pixels in the camera image.

    This camera model uses the conventions:

     * Top-left corner of the image is `(0, 0)`
     * Bottom-right corner is `(w, h)` (NOT `(h, w)`)
     * Pixels are centered, so `px_centers()[0, 0] == 0.5`

    Returns:
      2D image coordinates of center of all pixels of shape `(h, w, 2)`.
    """
    raise NotImplementedError

  def replace_fig_config(
      self,
      *,
      scale: float = FigConfig.scale,
  ) -> CameraSpec:
    """Returns a copy of self with figure params overwritten."""
    return self.replace(fig_config=self.fig_config.replace(scale=scale))

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
    corners_world = self.cam_from_px @ self.xnp.array(corners_px)
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


@dataclasses.dataclass(frozen=True)
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

    K = xnp.array([  # pylint: disable=invalid-name
        [focal_in_px, 0, cw],  # cx
        [0, focal_in_px, ch],  # cy
        [0, 0, 1],
    ])
    return cls(
        K=K, resolution=resolution, focal_length=(focal_in_px, focal_in_px)
    )

  @property
  @transformation.custom_transform
  def px_from_cam(  # pylint: disable=bad-staticmethod-argument,property-with-parameters
      self,
      points3d: DcOrArray,
  ) -> DcOrArray:
    points3d = dca.utils.np_utils.asarray(points3d, xnp=self.xnp)
    if isinstance(points3d, dca.DataclassArray):
      assert_supports_protocol(points3d, 'apply_px_from_cam')
      return points3d.apply_px_from_cam(self)
    else:
      return self._px_from_cam(points3d)

  @typing.overload
  def _px_from_cam(
      self,
      points3d: FloatArray['*d 3'],
      *,
      with_depth: bool = False,
  ) -> FloatArray['*d 2']:
    ...

  @typing.overload
  def _px_from_cam(
      self,
      points3d: FloatArray['*d 3'],
      *,
      with_depth: bool = True,
  ) -> tuple[FloatArray['*d 2'], FloatArray['*d 1']]:
    ...

  def _px_from_cam(
      self,
      points3d,
      *,
      with_depth: bool = False,
  ):
    """Project camera 3d coordinates to px 2d coordinates (internal).

    Args:
      points3d: 3d points
      with_depth: If `True`, also return the depth in a separated valiable

    Returns:
      `point2d`: If `with_depth=False`
      `(point2d, depth)`: If `with_depth=True`
    """
    if points3d.shape[-1] != 3:
      raise ValueError(f'Expected cam coords {points3d.shape} to be (..., 3)')

    # K @ [X,Y,Z] -> s * [u, v, 1]
    # (3, 3) @ (..., 3) -> (..., 3)
    points2d = self.xnp.einsum('ij,...j->...i', self.K, points3d)
    # Normalize: s * [u, v, 1] -> [u, v, 1]
    # And only keep [u, v]
    depth = points2d[..., 2:3]
    points2d = (points2d[..., :2] / (depth + 1e-8))

    if with_depth:
      return points2d, depth
    else:
      return points2d

  @property
  @transformation.custom_transform
  def cam_from_px(  # pylint: disable=bad-staticmethod-argument,property-with-parameters
      self,
      points2d: FloatArray['*d 2'],
  ) -> FloatArray['*d 3']:
    points2d = dca.utils.np_utils.asarray(points2d, xnp=self.xnp)
    if isinstance(points2d, dca.DataclassArray):
      assert_supports_protocol(points2d, 'apply_cam_from_px')
      return points2d.apply_cam_from_px(self)
    else:
      return self._cam_from_px(points2d)

  def _cam_from_px(  # pylint: disable=bad-staticmethod-argument,property-with-parameters
      self,
      points2d: FloatArray['*d 2'],
  ) -> FloatArray['*d 3']:
    assert not self.shape  # Should be vectorized
    points2d = dca.utils.np_utils.asarray(points2d, xnp=self.xnp)
    if points2d.shape[-1] != 2:
      raise ValueError(f'Expected pixel coords {points2d.shape} to be (..., 2)')

    # [u, v] -> [u, v, 1]
    # Concatenate (..., 2) with (..., 1) -> (..., 3)
    points2d = np_utils.append_row(points2d, 1., axis=-1)

    # [X,Y,Z] / s = K-1 @ [u, v, 1]
    # (3, 3) @ (..., 3) -> (..., 3)
    k_inv = enp.linalg.inv(self.K)
    points3d = self.xnp.einsum('ij,...j->...i', k_inv, points2d)

    # TODO(epot): Option to return normalized rays ?
    # Set z to -1
    # [X,Y,Z] -> [X, Y, Z=1]
    points3d = points3d / self.xnp.expand_dims(points3d[..., 2], axis=-1)
    return points3d

  @dca.vectorize_method
  def px_centers(self):
    xnp = self.xnp
    coord_w, coord_h = xnp.meshgrid(
        xnp.arange(self.w),  # w
        xnp.arange(self.h),  # h
        indexing='xy',
    )
    points2d = xnp.stack([coord_w, coord_h], axis=-1)
    points2d = xnp.asarray(points2d, dtype=xnp.float32) + .5
    assert points2d.shape == (self.h, self.w, 2)
    return points2d

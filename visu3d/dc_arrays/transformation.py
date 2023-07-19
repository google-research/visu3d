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

"""Transformation utils."""

from __future__ import annotations

import functools
from typing import Callable, Generic, TypeVar

import dataclass_array as dca
from dataclass_array.typing import DcT
import einops
from etils import enp
from etils.array_types import FloatArray  # pylint: disable=g-multiple-import
from visu3d import array_dataclass
from visu3d import math
from visu3d.dc_arrays import ray as ray_lib
from visu3d.utils import np_utils
from visu3d.utils import py_utils
from visu3d.utils.lazy_imports import plotly_base

_T = TypeVar('_T')


class TransformBase(array_dataclass.DataclassArray):
  """Transformation base class.

  Defines the transformation interface.

  """

  def __matmul__(self, other: _T) -> _T:
    """Apply the transformation."""
    raise NotImplementedError('Abstract method.')

  @property
  def inv(self: _T) -> _T:
    """Returns the inverse camera transform."""
    raise NotImplementedError('Abstract method.')


class Transform(TransformBase):
  """Affine transformation (Position, rotation and scale of an object).

  Attributes:
    R: Rotation/scale/skewing of the transformation ( `[[x0, y0, z0], [x1, y1,
      z1], [x2, y2, z2]]`)
    t: Translation of the transformation (`tx, ty, tz`)
  """

  R: FloatArray['*shape 3 3'] = (  # pylint: disable=invalid-name
      (1, 0, 0),
      (0, 1, 0),
      (0, 0, 1),
  )
  t: FloatArray['*shape 3'] = (0, 0, 0)

  @classmethod
  def identity(cls) -> Transform:
    """Returns the identity transform."""
    return cls(R=enp.lazy.np.eye(3), t=[0, 0, 0])

  @classmethod
  def from_matrix(cls, matrix: FloatArray['*shape 3 3']) -> Transform:
    """Constructs from a 4x4 transform matrix."""
    return cls(R=matrix[..., :3, :3], t=matrix[..., :3, 3])

  @classmethod
  def from_angle(cls, *, x=None, y=None, z=None) -> Transform:
    """Returns a transformation rotation around an axis (in radians).

    Example:

    ```python
    tr = v3d.Transform.from_angle(x=1/4 * enp.tau)  # Rotate 90° around x
    ```

    Rotations are applied following the Tait-Bryan chained rotations (z, y, x):

    ```python
    R = Rz @ Ry @ Rx
    ```

    See:
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_angles_(z-y%E2%80%B2-x%E2%80%B3_intrinsic)_%E2%86%92_rotation_matrix

    Args:
      x: Rotation around x (in radians): roll == ϕ == phi == x
      y: Rotation around y (in radians): pitch == θ == theta == y
      z: Rotation around z (in radians): yaw == ψ == psi == z

    Returns:
      tr: The transformation.
    """
    return cls(R=math.euler_to_rot(x=x, y=y, z=z))

  @classmethod
  @enp.check_and_normalize_arrays(strict=False)
  def from_look_at(
      cls,
      *,
      pos: FloatArray['*shape 3'],
      target: FloatArray['*shape 3'],
  ) -> Transform:
    """Factory to create a transformation which look at.

    Used to convert camera to world coordinates.

    This transformation assume the following conventions:

    * World coordinates: Floor is (x, y), z pointing upward
    * Camera coordinates: See `v3d.CameraSpec` docstring.

    The transformation assume the `width` dimension of the camera is parallel
    to the floor of the world.

    Args:
      pos: Origin position
      target: Target position

    Returns:
      The Transform.
    """
    return cls(
        t=pos,
        R=_get_r_look_at_(pos=pos, target=target),
    )

  @dca.vectorize_method
  def look_at(self, target: FloatArray['*shape 3']) -> Transform:
    """Returns a new transform looking at the target."""
    target = dca.utils.np_utils.asarray(
        target,
        xnp=self.xnp,
        dtype=enp.lazy.np.float32,
    )
    # TODO(epot): Rather than overwriting R, should only apply the rotation
    # to the existing R.
    return self.replace(R=_get_r_look_at_(pos=self.t, target=target))

  @property
  @dca.vectorize_method
  def x_dir(self) -> FloatArray['*shape 3']:
    """`x` axis of the transformation (`[x0, x1, x2]`)."""
    return self.R[:, 0]  # pylint: disable=invalid-sequence-index

  @property
  @dca.vectorize_method
  def y_dir(self) -> FloatArray['*shape 3']:
    """`y` axis of the transformation (`[y0, y1, y2]`)."""
    return self.R[:, 1]  # pylint: disable=invalid-sequence-index

  @property
  @dca.vectorize_method
  def z_dir(self) -> FloatArray['*shape 3']:
    """`z` axis of the transformation (`[z0, z1, z2]`)."""
    return self.R[:, 2]  # pylint: disable=invalid-sequence-index

  @property
  @dca.vectorize_method
  def x_ray(self) -> ray_lib.Ray:
    """Array pointing to `z`."""
    return ray_lib.Ray(pos=self.t, dir=self.x_dir)

  @property
  @dca.vectorize_method
  def y_ray(self) -> ray_lib.Ray:
    """Array pointing to `z`."""
    return ray_lib.Ray(pos=self.t, dir=self.y_dir)

  @property
  @dca.vectorize_method
  def z_ray(self) -> ray_lib.Ray:
    """Array pointing to `z`."""
    return ray_lib.Ray(pos=self.t, dir=self.z_dir)

  @property
  @dca.vectorize_method
  def ray_basis(self) -> ray_lib.Ray:
    """The `(x, y, z)` basis of the transformation, as ray."""
    return ray_lib.Ray(
        # We can use `np` for the display
        pos=self.xnp.broadcast_to(self.t, self.shape + (3, 3)),
        # R is [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]] so we transpose
        # so dir is [[x0, x1, x2], [y0, y1, y2], [z0, z1, z2]]
        dir=self.xnp.asarray(self.R.T),
    )

  @property
  @dca.vectorize_method
  def scale_xyz(self) -> FloatArray['*shape 3']:
    """Returns the `(sx, sy, sz)` scale of the transform along each axis."""
    return enp.compat.norm(self.R, axis=0)

  @property
  @dca.vectorize_method
  def scale(self):
    """Returns the global scale (if `x, y, z` share the same scale).

    Returns:
      The global scale shared by all axis.

    Raises:
      ValueError: If the `x, y, z` scales are different between axis.
    """

    def _err_msg():
      return (
          'Cannot get `Transform.scale` when x, y, z scale are '
          f'different: {self.scale_xyz}'
      )

    xnp = self.xnp
    scale_xyz = xnp.round(self.scale_xyz, decimals=7)
    # TODO(epot): Move into `enp.ops.unique`
    if enp.lazy.is_np_xnp(xnp):
      global_scales = xnp.unique(scale_xyz)
      raise_error = len(global_scales) != 1
    elif enp.lazy.is_jax_xnp(xnp):
      global_scales, global_count = xnp.unique(
          scale_xyz,
          size=1,
          return_counts=True,
      )

      # Unfortunately, Jax don't have an easy API to conditionally
      # raise error within a `jax.jit` function
      # This won't have any effect when the function is traced.
      from jax.experimental import checkify  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top

      checkify.check(global_count[0] == 3, msg=_err_msg())

      raise_error = False
    elif enp.lazy.is_torch_xnp(xnp):
      global_scales = enp.lazy.torch.unique(scale_xyz)
      raise_error = len(global_scales) != 1
    elif enp.lazy.is_tf_xnp(xnp):
      global_scales, _ = enp.lazy.tf.unique(scale_xyz)
      raise_error = len(global_scales) != 1
    else:
      raise AssertionError(f'Unknown numpy module {xnp}')
    if raise_error:
      raise ValueError(_err_msg())
    else:
      (global_scale,) = global_scales
    return global_scale

  def mul_scale(self, factor: FloatArray['*shape 3?']) -> Transform:
    """Scale the transformation `(sx, sy, sz)` by the given factor.

    This is similar to `v3d.Ray.scale_dir`, applied on each individual axis.

    Args:
      factor: The factor by which scale each axis. If scalar, broadcast all axis
        by the same value.

    Returns:
      The new transform with scaled.
    """
    xnp = self.xnp
    factor = xnp.asarray(factor)
    if factor.shape == ():  # pylint: disable=g-explicit-bool-comparison
      new_r = self.R * factor
    elif factor.shape == (3,):
      new_r = self.R @ xnp.diag(factor)
    else:
      raise ValueError(f'Unsupported factor shape {factor.shape}')
    return self.replace(R=new_r)

  @dca.vectorize_method
  def normalize(self) -> Transform:
    """Normalize the scale x, y, z to be `(1, 1, 1)`."""
    xnp = self.xnp
    return self.replace(R=self.R @ xnp.diag(1 / self.scale_xyz))

  @property
  @dca.vectorize_method
  def matrix4x4(self) -> FloatArray['*shape 4 4']:
    """Returns the 4x4 transformation matrix.

    [R|t]
    [0|1]
    """
    t = einops.rearrange(self.t, '... d -> ... d 1')
    matrix3x4 = self.xnp.concatenate([self.R, t], axis=-1)
    assert matrix3x4.shape == (3, 4)
    last_row = self.xnp.asarray([[0, 0, 0, 1]])
    return self.xnp.concatenate([matrix3x4, last_row], axis=-2)

  @property
  @dca.vectorize_method
  def inv(self) -> Transform:
    """Returns the inverse camera transform."""
    # Might be a more optimized way than stacking/unstacking matrix
    return type(self).from_matrix(enp.compat.inv(self.matrix4x4))

  def __add__(self, translation: FloatArray['... 3']) -> Transform:
    """Translate the position."""
    translation = self.xnp.asarray(translation)
    return self.replace(t=self.t + translation)

  __sub__ = np_utils.__sub__

  @dca.vectorize_method
  def __matmul__(self, other: _T) -> _T:
    """Apply the transformation the array or dataclass array.

    Transformation & inputs can be arbitrarly batched. Transform will be
    applied on individual elements (vectorized, NOT broadcasted):

    For example, for `tr @ other`, the shape are vectorized using the
    `dca.vectorize_method` rules:

    ```python
    () @ (*x,) -> (*x,)
    (*tr,) @ (*tr, *x) -> (*tr, *x)
    ```

    Any `v3d.DataclassArray` can support `v3d.Transform` by implementing
    the `apply_transform` protocol:

    ```python
    my_obj = tr @ my_obj  # Call `my_obj.apply_transform(tr)`
    ```

    Inside the `apply_transform` function, `tr.shape == ()`. Vectorization is
    auto-supported.

    Args:
      other: The array or dataclass array on which apply the transformation.
        Arrays are interpreted as 3d point cloud so should be `Array[..., 3]`.
        `v3d.DataclassArray` should implement the `apply_transform` protocol.

    Returns:
      The new `other` object after transformation.
    """
    self.assert_same_xnp(other)
    if enp.lazy.is_array(other):
      return self.apply_to_pos(other)
    elif isinstance(other, dca.DataclassArray):
      py_utils.assert_supports_protocol(
          other,
          'apply_transform',
          msg=f'`v3d.Transform` @ `{type(other).__name__}` not supported.',
      )
      return other.apply_transform(self)
    else:
      raise TypeError(f'Unexpected type: {type(other)}')

  # TODO(epot): Also add a `tr.apply_to` method which supports broadcasting

  @dca.vectorize_method
  def apply_to_pos(self, point: FloatArray['*d 3']) -> FloatArray['*d 3']:
    """Apply the transformation on the point cloud."""
    self.assert_same_xnp(point)
    point = self.xnp.asarray(point)
    if point.shape[-1] != 3:
      raise ValueError(f'point shape should be `(..., 3)`. Got {point.shape}')
    return self.apply_to_dir(point) + self.t

  @dca.vectorize_method
  def apply_to_dir(
      self,
      direction: FloatArray['*d 3'],
  ) -> FloatArray['*d 3']:
    """Apply the transformation on the direction."""
    self.assert_same_xnp(direction)
    # Direction are invariant to translation
    return self.xnp.einsum('ij,...j->...i', self.R, direction)

  # Protocols (inherited)

  # Apply transform is the protocol implementation but is NOT part of the public
  # API, so don't have `@dca.vectorize_method`
  def apply_transform(
      self,
      tr: Transform,
  ) -> Transform:
    return self.replace(
        R=tr.R @ self.R,
        t=tr.apply_to_pos(self.t),
    )

  def make_traces(self) -> list[plotly_base.BaseTraceType]:
    rays = self.ray_basis
    (line_trace,) = rays.make_traces()
    line_trace.mode = 'lines+markers+text'
    # Add x, y, z text labels to the plot
    # Each point in the original line trace is:
    # [ray_origin, ray_end, None(=line break)]
    # So we only add the text on the `ray_end`
    # fmt: off
    line_trace.text = [
        None, 'x', None,
        None, 'y', None,
        None, 'z', None,
    ] * rays.size
    # fmt: on
    return [line_trace]


# TODO(epot): Have custom transform support `.inv`
#
# @cam_from_px.register_inv
# def px_from_cam(self, point3d):
#


def custom_transform(
    fn: Callable[[DcT, _T], _T],
) -> Callable[[DcT], CustomTransform[DcT, _T]]:
  """Custom transformation decorator.

  This decorator is a wrapper around custom transformations method to allow
  composing custom transformations with `v3d.Transform`.

  Usage:

  ```python
  class MyObject(v3d.DataclassArray):

    @property
    @v3d.custom_transform
    def my_transform(self, points: _T) -> _T:
      ...  # Custom transform implementation
      return points

  obj = MyObject()

  # The custom transform can be composed with other
  points = obj.my_transform @ points
  points = obj.my_transform @ cam.world_from_px @ px_coords
  ```

  Note: The decorated method is automatically vectorized. See
  `dca.vectorize_method`.

  Args:
    fn: Method to decorate

  Returns:
    The decorated function (which is should be decorated with @property).
  """

  @functools.wraps(fn)
  def decorated(self) -> CustomTransform:
    return CustomTransform(self_=self, method=fn)

  return decorated


class CustomTransform(TransformBase, Generic[DcT, _T]):  # pytype: disable=invalid-function-definition
  """Custom transformation method wrapper.

  Attributes:
    self_: Object from which the transform is applied.
    method: Transform method (decorated by `@v3d.custom_transform`). Calling
      `custom_tr @ x` is equivalent of calling `self.method(x)`
  """

  self_: DcT = dca.field(  # pytype: disable=annotation-type-mismatch
      shape=(),
      dtype=dca.DataclassArray,
  )
  # It is important that:
  # * method should NOT be a lambda (to avoid accidental closure side effects
  # * method should NOT be a bound method (to not bind `self`)
  # TODO(epot): Add `__post_init__` check for this.
  method: Callable[[DcT, _T], _T]

  @dca.vectorize_method
  def __matmul__(self, other: _T) -> _T:
    if isinstance(other, TransformBase):
      # If other is another transform: create a composed transform
      return self.compose_transform(other)
    else:
      return self.method(self.self_, other)

  def compose_transform(self, other_tr: TransformBase) -> TransformBase:
    """Called when `tr_custom @ tr`."""
    return CustomTransform(
        self_=ComposedTransform(
            left_tr=self,
            right_tr=other_tr,
        ),
        method=ComposedTransform._apply_to,  # pylint: disable=protected-access
    )

  def apply_transform(self, tr: Transform) -> DcT:
    """Called when `tr @ tr_custom`."""
    return CustomTransform(
        self_=ComposedTransform(
            left_tr=tr,
            right_tr=self,
        ),
        method=ComposedTransform._apply_to,  # pylint: disable=protected-access
    )


class ComposedTransform(TransformBase):
  """Transform composed to 2 transformations.

  Allow to chain arbitrary custom transforms:

  ```
  px = px_from_cam @ cam_from_world @ world_points
  ```

  Note: Should not be created directly but only inside `CustomTransform`.

  """

  left_tr: TransformBase
  right_tr: TransformBase

  def _apply_to(self: ComposedTransform, other: _T) -> _T:
    """Apply composed method."""
    return self.left_tr @ (self.right_tr @ other)


def _get_r_look_at_(
    *,
    pos: FloatArray['*shape 3'],
    target: FloatArray['*shape 3'],
) -> FloatArray['*shape 3 3']:
  """Compute the `R` (3, 3) matrix."""
  # TODO(epot): Support more conventions
  # * `up='z'`
  # * `mode='camera'` (x==h pointing ↓, y==w pointing →) or right hand rule.

  xnp = enp.lazy.get_xnp(pos)
  _assert_shape(pos, 'pos')
  _assert_shape(target, 'target')

  cam_forward = enp.linalg.normalize(target - pos)

  # In world coordinates, `z` is pointing up
  world_up = xnp.asarray([0, 0, 1.0], dtype=xnp.float32)
  # The width of the cam is parallel to the ground (prependicular to z), so
  # use cross-product.
  cam_w = xnp.cross(cam_forward, world_up)
  cam_w = enp.linalg.normalize(cam_w)

  # Similarly, the height is pointing downward.
  cam_h = xnp.cross(cam_forward, cam_w)

  R = xnp.stack([cam_w, cam_h, cam_forward], axis=-1)  # pylint: disable=invalid-name
  return R


def _assert_shape(array: FloatArray['*d 4'], name: str) -> None:
  """Test that array shape end by 3."""
  if array.shape[-1] != 3:
    raise ValueError(f'{name!r} shape should end be (3,). Got {array.shape}')

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

"""Dataclass array."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Callable, Generic, Iterator, Optional, Tuple, Type, TypeVar, Union

import einops
from etils import array_types
from etils import edc
from etils import enp
from etils import epy
from etils.array_types import Array
import numpy as np
from typing_extensions import Literal, TypeAlias  # pylint: disable=g-multiple-import
from visu3d import shape_parsing
from visu3d import type_parsing
from visu3d.plotly import fig_utils
from visu3d.typing import Axes, DcOrArray, DcOrArrayT, DTypeArg, Shape  # pylint: disable=g-multiple-import
from visu3d.utils import np_utils
from visu3d.utils import py_utils

if typing.TYPE_CHECKING:
  from visu3d.dc_arrays import transformation

lazy = enp.lazy

# TODO(pytype): Should use `v3d.typing.DcT` but bound does not work across
# modules.
_DcT = TypeVar('_DcT', bound='DataclassArray')

# Any valid numpy indices slice ([x], [x:y], [:,...], ...)
_IndiceItem = Union[type(Ellipsis), None, int, slice, Any]
_Indices = Tuple[_IndiceItem]  # Normalized slicing
_IndicesArg = Union[_IndiceItem, _Indices]

_METADATA_KEY = 'v3d_field'


class DataclassArray(fig_utils.Visualizable):
  """Dataclass which behaves like an array.

  Usage:

  ```python
  @dataclasses.dataclass
  class Square(DataclassArray):
    pos: f32['*shape 2']
    scale: f32['*shape']
    name: str

  # Create 3 squares batched
  p = Square(
      pos=[[x0, y0], [x1, y1], [x2, y2]],
      scale=[scale0, scale1, scale2],
      name='my_square',
  )
  p.shape == (3,)
  p.pos.shape == (3, 2)
  p[0] == Square(pos=[x0, y0], scale=scale0)

  p = p.reshape((3, 1))  # Reshape the inner-shape
  p.shape == (3, 1)
  p.pos.shape == (3, 1, 2)

  p.name == 'my_square'
  ```

  `DataclassArray` has 2 types of fields:

  * Array fields: Fields batched like numpy arrays, with reshape, slicing,...
    (`pos` and `scale` in the above example).
  * Static fields: Other non-numpy field. Are not modified by reshaping,... (
    `name` in the above example).
    Static fields are also ignored in `jax.tree_map`.

  `DataclassArray` detect array fields if either:

  * The typing annotation is a `etils.array_types` annotation (in which
    case shape/dtype are automatically infered from the typing annotation)
    Example: `x: f32[..., 3]`
  * The typing annotation is another `v3d.DataclassArray` (in which case
    `my_dataclass.field.shape == my_dataclass.shape`)
    Example: `x: MyDataclass`
  * The field is explicitly defined in `v3d.array_field`, in which case
    the typing annotation is ignored.
    Example: `x: Any = v3d.field(shape=(), dtype=np.int64)`

  Field which do not satisfy any of the above conditions are static (including
  field annotated with `field: np.ndarray` or similar).

  """
  _shape: Shape
  _xnp: enp.NpModule

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # TODO(epot): Could have smart __repr__ which display types if array have
    # too many values.
    edc.dataclass(kw_only=True, repr=True)(cls)
    cls._v3d_tree_map_registered = False

  def __post_init__(self) -> None:
    """Validate and normalize inputs."""
    cls = type(self)

    # Make sure the dataclass was registered and frozen
    if not dataclasses.is_dataclass(cls) or not cls.__dataclass_params__.frozen:  # pytype: disable=attribute-error
      raise ValueError(
          '`v3d.DataclassArray` need to be @dataclasses.dataclass(frozen=True)')

    # Register the tree_map here instead of `__init_subclass__` as `jax` may
    # not have been registered yet during import
    if enp.lazy.has_jax and not cls._v3d_tree_map_registered:  # pylint: disable=protected-access
      enp.lazy.jax.tree_util.register_pytree_node_class(cls)
      cls._v3d_tree_map_registered = True  # pylint: disable=protected-access

    # Note: Calling the `_all_array_fields` property during `__init__` will
    # normalize the arrays (`list` -> `np.ndarray`). This is done in the
    # `_ArrayField` contructor
    if not self._all_array_fields:
      raise ValueError(
          f'{self.__class__.__qualname__} should have at least one '
          '`v3d.array_field`')

    # Validate the array type is consistent and cast np -> xnp
    xnp = self._cast_dtype_inplace()

    # Validate the batch shape is consistent
    # Because this is only done inside `__init__`, it should be ok to
    # mutate self.
    # However, we need to be careful that `_ArrayField` never uses
    # `@epy.cached_property`
    shape = self._broadcast_shape_inplace()

    if xnp is None:  # No values
      # Inside `jax.tree_utils`, tree-def can be created with `None` values.
      assert shape is None
      xnp = np

    # Cache results
    # Should the state be stored in a separate object to avoid collisions ?
    assert shape is None or isinstance(shape, tuple), shape
    self._setattr('_shape', shape)
    self._setattr('_xnp', xnp)

  # ====== Array functions ======

  @property
  def shape(self) -> Shape:
    """Returns the batch shape common to all fields."""
    return self._shape

  @property
  def size(self) -> int:
    """Returns the number of elements."""
    return np_utils.size_of(self._shape)

  @property
  def ndim(self) -> int:
    """Returns the number of dimensions."""
    return len(self._shape)

  def reshape(self: _DcT, shape: Union[Shape, str], **axes_length: int) -> _DcT:
    """Reshape the batch shape according to the pattern.

    Supports both tuple and einops mode:

    ```python
    rays.reshape('b h w -> b (h w)')
    rays.reshape((128, -1))
    ```

    Args:
      shape: Target shape. Can be string for `einops` support.
      **axes_length: Any additional specifications for dimensions for einops
        support.

    Returns:
      The dataclass array with the new shape
    """
    if isinstance(shape, str):  # Einops support
      return self._map_field(
          array_fn=lambda f: einops.rearrange(  # pylint: disable=g-long-lambda
              f.value,
              np_utils.to_absolute_einops(shape, nlastdim=len(f.inner_shape)),
              **axes_length,
          ),
          dc_fn=lambda f: f.value.reshape(  # pylint: disable=g-long-lambda
              np_utils.to_absolute_einops(shape, nlastdim=len(f.inner_shape)),
              **axes_length,
          ),
      )
    else:  # Numpy support
      assert isinstance(shape, tuple)  # For pytest

      def _reshape(f: _ArrayField):
        return f.value.reshape(shape + f.inner_shape)

      return self._map_field(array_fn=_reshape, dc_fn=_reshape)

  def flatten(self: _DcT) -> _DcT:
    """Flatten the batch shape."""
    return self.reshape((-1,))

  def broadcast_to(self: _DcT, shape: Shape) -> _DcT:
    """Broadcast the batch shape."""
    # pyformat: disable
    return self._map_field(
        array_fn=lambda f: f.broadcast_to(shape),
        dc_fn=lambda f: f.broadcast_to(shape),
    )
    # pyformat: enable

  def __getitem__(self: _DcT, indices: _IndicesArg) -> _DcT:
    """Slice indexing."""
    indices = np.index_exp[indices]  # Normalize indices
    # Replace `...` by explicit shape
    indices = _to_absolute_indices(indices, shape=self.shape)
    return self._map_field(
        array_fn=lambda f: f.value[indices],
        dc_fn=lambda f: f.value[indices],
    )

  # _DcT[n *d] -> Iterator[_DcT[*d]]
  def __iter__(self: _DcT) -> Iterator[_DcT]:
    """Iterate over the outermost dimension."""
    if not self.shape:
      raise TypeError(f'iteration over 0-d array: {self!r}')

    # Similar to `etree.unzip(self)` (but work with any backend)
    field_names = [f.name for f in self._array_fields]  # pylint: disable=not-an-iterable
    field_values = [f.value for f in self._array_fields]  # pylint: disable=not-an-iterable
    for vals in zip(*field_values):
      yield self.replace(**dict(zip(field_names, vals)))

  def __len__(self) -> int:
    """Length of the first array dimension."""
    if not self.shape:
      raise TypeError(
          f'len() of unsized {self.__class__.__name__} (shape={self.shape})')
    return self.shape[0]

  def __bool__(self) -> Literal[True]:
    """`v3d.DataclassArray` always evaluate to `True`.

    Like all python objects (including dataclasses), `v3d.DataclassArray` always
    evaluate to `True`. So:
    `Ray(pos=None)`, `Ray(pos=0)` all evaluate to `True`.

    This allow construct like:

    ```python
    def fn(ray: Optional[v3d.Ray] = None):
      if ray:
        ...
    ```

    Or:

    ```python
    def fn(ray: Optional[v3d.Ray] = None):
      ray = ray or default_ray
    ```

    Only in the very rare case of empty-tensor (`shape=(0, ...)`)

    ```python
    assert ray is not None
    assert len(ray) == 0
    bool(ray)  # TypeError: Truth value is ambigous
    ```

    Returns:
      True

    Raises:
      ValueError: If `len(self) == 0` to avoid ambiguity.
    """
    if self.shape and not len(self):  # pylint: disable=g-explicit-length-test
      raise ValueError(
          f'The truth value of {self.__class__.__name__} when `len(x) == 0` '
          'is ambigous. Use `len(x)` or `x is not None`.')
    return True

  def map_field(
      self: _DcT,
      fn: Callable[[Array['*din']], Array['*dout']],
  ) -> _DcT:
    """Apply a transformation on all arrays from the fields."""
    return self._map_field(
        array_fn=lambda f: fn(f.value),
        dc_fn=lambda f: f.value.map_field(fn),
    )

  # ====== Dataclass/Conversion utils ======

  # TODO(pytype): Could be removed once there's a way of annotating this.
  replace = edc.dataclass_utils.replace

  def as_np(self: _DcT) -> _DcT:
    """Returns the instance as containing `np.ndarray`."""
    return self.as_xnp(enp.lazy.np)

  def as_jax(self: _DcT) -> _DcT:
    """Returns the instance as containing `jnp.ndarray`."""
    return self.as_xnp(enp.lazy.jnp)

  def as_tf(self: _DcT) -> _DcT:
    """Returns the instance as containing `tf.Tensor`."""
    return self.as_xnp(enp.lazy.tnp)

  def as_xnp(self: _DcT, xnp: enp.NpModule) -> _DcT:
    """Returns the instance as containing `xnp.ndarray`."""
    if xnp is self.xnp:  # No-op
      return self
    return self.map_field(xnp.asarray)

  # ====== Internal ======

  # TODO(pytype): Remove hack. Currently, Python does not support typing
  # annotations for modules, by pytype auto-infer the correct type.
  # So this hack allow auto-completion

  if typing.TYPE_CHECKING:

    @property
    def xnp(self):  # pylint: disable=function-redefined
      """Returns the numpy module of the class (np, jnp, tnp)."""
      return np

  else:

    @property
    def xnp(self) -> enp.NpModule:
      """Returns the numpy module of the class (np, jnp, tnp)."""
      return self._xnp

  @epy.cached_property
  def _all_array_fields(self) -> dict[str, _ArrayField]:
    """All array fields, including `None` values."""
    # Validate and normalize array fields (e.g. list -> np.array,...)
    # At this point, `ForwardRef` should have been resolved.
    hints = typing.get_type_hints(type(self))
    # TODO(py3.8):
    # return {  # pylint: disable=g-complex-comprehension
    #     f.name: array_field
    #     for f in dataclasses.fields(self)
    #     if (array_field := _make_array_field(self, f, hints)) is not None
    # }
    out = {}
    for f in dataclasses.fields(self):
      array_field_ = _make_array_field(self, f, hints)
      if array_field_ is None:
        continue
      out[f.name] = array_field_
    return out

  @epy.cached_property
  def _array_fields(self) -> list[_ArrayField]:
    """All active array fields (non-None)."""
    # Filter `None` values
    return [
        f for f in self._all_array_fields.values() if not f.is_value_missing
    ]

  def _cast_dtype_inplace(self) -> Optional[enp.NpModule]:
    """Validate `xnp` are consistent and cast `np` -> `xnp` in-place."""
    if not self._array_fields:  # No fields have been defined.
      return None

    xnps = py_utils.groupby(
        self._array_fields,
        key=lambda f: f.xnp,
        value=lambda f: f.name,
    )
    if not xnps:
      return None
    xnp = _infer_xnp(xnps)

    def _cast_field(f: _ArrayField) -> None:
      if f.xnp is xnp:
        return
      self._setattr(f.name, np_utils.asarray(f.value, xnp=xnp))

    self._map_field(
        array_fn=_cast_field,
        dc_fn=_cast_field,  # pytype: disable=wrong-arg-types
    )
    return xnp

  def _broadcast_shape_inplace(self) -> Optional[Shape]:
    """Validate the shapes are consistent and broadcast values in-place."""
    if not self._array_fields:  # No fields have been defined.
      return None

    # First collect all shapes and compute the final shape.
    shape_to_names = py_utils.groupby(
        self._array_fields,
        key=lambda f: f.host_shape,
        value=lambda f: f.name,
    )
    shape_lengths = {len(s) for s in shape_to_names.keys()}

    # Broadcast all shape together
    try:
      final_shape = np.broadcast_shapes(*shape_to_names.keys())
    except ValueError:
      final_shape = None  # Bad broadcast

    # Currently, we restrict broadcasting to either scalar or fixed length.
    # This is to avoid confusion broadcasting vs vectorization rules.
    # This restriction could be lifted if we encounter a use-case.
    # pyformat: disable
    if (
        final_shape is None
        or len(shape_lengths) > 2
        or (len(shape_lengths) == 2 and 0 not in shape_lengths)
    ):
      # pyformat: enable
      raise ValueError(
          f'Conflicting batch shapes: {shape_to_names}. '
          f'Currently {type(self).__qualname__}.__init__ broadcasting is '
          'restricted to scalar or dim=1 . '
          'Please open an issue if you need more fine-grained broadcasting.')

    def _broadcast_field(f: _ArrayField) -> None:
      if f.host_shape == final_shape:  # Already broadcasted
        return
      self._setattr(f.name, f.broadcast_to(final_shape))

    self._map_field(
        array_fn=_broadcast_field,
        dc_fn=_broadcast_field,  # pytype: disable=wrong-arg-types
    )
    return final_shape

  def _to_absolute_axis(self, axis: Axes) -> Axes:
    """Normalize the axis to absolute value."""
    try:
      return np_utils.to_absolute_axis(axis, ndim=self.ndim)
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(
          e,
          prefix=f'For {self.__class__.__qualname__} with shape={self.shape}: ',
      )

  def _map_field(
      self: _DcT,
      *,
      array_fn: Callable[[_ArrayField[Array['*din']]], Array['*dout']],
      dc_fn: Optional[Callable[[_ArrayField[_DcT]], _DcT]],
  ) -> _DcT:
    """Apply a transformation on all array fields structure.

    Args:
      array_fn: Function applied on the `xnp.ndarray` fields
      dc_fn: Function applied on the `v3d.DataclassArray` fields (to recurse)

    Returns:
      The transformed dataclass array.
    """

    def _apply_field_dn(f: _ArrayField):
      if f.is_dataclass:  # Recurse on dataclasses
        return dc_fn(f)  # pylint: disable=protected-access
      else:
        return array_fn(f)

    new_values = {f.name: _apply_field_dn(f) for f in self._array_fields}  # pylint: disable=not-an-iterable
    return self.replace(**new_values)

  def tree_flatten(self) -> tuple[list[DcOrArray], _TreeMetadata]:
    """`jax.tree_utils` support."""
    # We flatten all values (and not just the non-None ones)
    array_field_values = [f.value for f in self._all_array_fields.values()]
    metadata = _TreeMetadata(
        array_field_names=list(self._all_array_fields.keys()),
        non_array_field_kwargs={
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in self._all_array_fields  # pylint: disable=unsupported-membership-test
        },
    )
    return (array_field_values, metadata)

  @classmethod
  def tree_unflatten(
      cls: Type[_DcT],
      metadata: _TreeMetadata,
      array_field_values: list[DcOrArray],
  ) -> _DcT:
    """`jax.tree_utils` support."""
    array_field_kwargs = dict(
        zip(
            metadata.array_field_names,
            array_field_values,
        ))
    init_fields = {}
    non_init_fields = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for k, v in metadata.non_array_field_kwargs.items():
      if fields[k].init:
        init_fields[k] = v
      else:
        non_init_fields[k] = v

    self = cls(**array_field_kwargs, **init_fields)
    # Currently it's not clear how to handle non-init fields so raise an error
    if non_init_fields:
      if set(non_init_fields) != {'fig_config'}:
        raise ValueError(
            '`v3d.DataclassArray` with init=False field not supported yet.')
      # TODO(py3.10): Delete once dataclass supports `kw_only=True`
      self._setattr('fig_config', non_init_fields['fig_config'])  # pylint: disable=protected-access
    return self

  def _setattr(self, name: str, value: Any) -> None:
    """Like setattr, but support `frozen` dataclasses."""
    object.__setattr__(self, name, value)

  def assert_same_xnp(self, x: Union[Array[...], DataclassArray]) -> None:
    """Assert the given array is of the same type as the current object."""
    xnp = np_utils.get_xnp(x)
    if xnp is not self.xnp:
      raise ValueError(
          f'{self.__class__.__name__} is {self.xnp.__name__} but got input '
          f'{xnp.__name__}. Please cast input first.')


def _infer_xnp(xnps: dict[enp.NpModule, list[str]]) -> enp.NpModule:
  """Extract the `xnp` module."""
  non_np_xnps = set(xnps) - {np}  # jnp, tnp take precedence on `np`

  # Detecting conflicting xnp
  if len(non_np_xnps) > 1:
    xnps = {k.__name__: v for k, v in xnps.items()}
    raise ValueError(f'Conflicting numpy types: {xnps}')

  if not non_np_xnps:
    return np
  else:
    (xnp,) = non_np_xnps
    return xnp


def _count_not_none(indices: _Indices) -> int:
  """Count the number of non-None and non-ellipsis elements."""
  return len([k for k in indices if k is not np.newaxis and k is not Ellipsis])


def _count_ellipsis(elems: _Indices) -> int:
  """Returns the number of `...` in the indices."""
  # Cannot use `elems.count(Ellipsis)` because `np.array() == Ellipsis` fail
  return len([elem for elem in elems if elem is Ellipsis])


def _to_absolute_indices(indices: _Indices, *, shape: Shape) -> _Indices:
  """Normalize the indices to replace `...`, by `:, :, :`."""
  assert isinstance(indices, tuple)
  ellipsis_count = _count_ellipsis(indices)
  if ellipsis_count > 1:
    raise IndexError("an index can only have a single ellipsis ('...')")
  valid_count = _count_not_none(indices)
  if valid_count > len(shape):
    raise IndexError(f'too many indices for array. Batch shape is {shape}, but '
                     f'rank-{valid_count} was provided.')
  if not ellipsis_count:
    return indices
  ellipsis_index = indices.index(Ellipsis)
  start_elems = indices[:ellipsis_index]
  end_elems = indices[ellipsis_index + 1:]
  ellipsis_replacement = [slice(None)] * (len(shape) - valid_count)
  return (*start_elems, *ellipsis_replacement, *end_elems)


@dataclasses.dataclass(frozen=True)
class _TreeMetadata:
  """Metadata forwarded in ``."""
  array_field_names: list[str]
  non_array_field_kwargs: dict[str, Any]


def array_field(
    shape: Shape,
    dtype: DTypeArg = float,
    **field_kwargs,
) -> dataclasses.Field[DcOrArray]:
  """Dataclass array field.

  See `v3d.DataclassArray` for example.

  Args:
    shape: Inner shape of the field
    dtype: Type of the field
    **field_kwargs: Args forwarded to `dataclasses.field`

  Returns:
    The dataclass field.
  """
  # TODO(epot): Validate shape, dtype
  v3d_field = _ArrayFieldMetadata(
      inner_shape=shape,
      dtype=dtype,
  )
  return dataclasses.field(**field_kwargs, metadata={_METADATA_KEY: v3d_field})


@edc.dataclass
@dataclasses.dataclass
class _ArrayFieldMetadata:
  """Metadata of the array field (shared across all instances).

  Attributes:
    inner_shape: Inner shape
    dtype: Type of the array. Can be `int`, `float`, `np.dtype` or
      `v3d.DataclassArray` for nested arrays.
  """
  inner_shape: Shape
  dtype: DTypeArg

  def __post_init__(self):
    """Normalizing/validating the shape/dtype."""
    # Validate shape
    self.inner_shape = tuple(self.inner_shape)
    if None in self.inner_shape:
      raise ValueError(f'Shape should be defined. Got: {self.inner_shape}')

    # Validate dtype
    if not self.is_dataclass:
      self.dtype = _validate_dtype(self.dtype)

  def to_dict(self) -> dict[str, Any]:
    """Returns the dict[field_name, field_value]."""
    return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

  @property
  def is_dataclass(self) -> bool:
    """Returns `True` if the field is a dataclass."""
    # Need to check `type` first as `issubclass` fails for `np.dtype('int32')`
    dtype = self.dtype
    return isinstance(dtype, type) and issubclass(dtype, DataclassArray)


@edc.dataclass
@dataclasses.dataclass
class _ArrayField(_ArrayFieldMetadata, Generic[DcOrArrayT]):
  """Array field of a specific dataclass instance.

  Attributes:
    name: Instance of the attribute
    host: Dataclass instance who this field is attached too
  """
  name: str
  host: DataclassArray

  def __post_init__(self):
    if self.is_value_missing:  # No validation when there is no value
      return
    if self.is_dataclass:
      self._init_dataclass()
    else:
      self._init_array()

    # Common assertions to all fields types
    if self.host_shape + self.inner_shape != self.value.shape:
      raise ValueError(
          f'Expected {type(self.host).__name__}.{self.name} shape to be '
          f'{(py_utils.Ellipsis, *self.inner_shape)}. Got: {self.value.shape}')

  @property
  def xnp(self) -> enp.NpModule:
    """Numpy module of the field."""
    return np_utils.get_xnp(self.value)

  def _init_array(self) -> None:
    """Initialize when the field is an array."""
    if isinstance(self.value, DataclassArray):
      raise TypeError(
          f'{self.name} should be {self.dtype}. Got: {type(self.value)}')
    # Convert and normalize the array
    xnp = lazy.get_xnp(self.value, strict=False)
    value = xnp.asarray(self.value, dtype=self.dtype)
    self.host._setattr(self.name, value)  # pylint: disable=protected-access

  def _init_dataclass(self) -> None:
    """Initialize when the field is a nested dataclass array."""
    if not isinstance(self.value, self.dtype):
      raise TypeError(
          f'{self.name} should be {self.dtype}. Got: {type(self.value)}')

  @property
  def value(self) -> DcOrArrayT:
    """Access the `host.<field-name>`."""
    return getattr(self.host, self.name)

  @property
  def is_value_missing(self) -> bool:
    """Returns `True` if the value wasn't set."""
    if self.value is None:
      return True
    elif type(self.value) is object:  # pylint: disable=unidiomatic-typecheck
      # Checking for `object` is a hack required for `@jax.vmap` compatibility:
      # In `jax/_src/api_util.py` for `flatten_axes`, jax set all values to a
      # dummy sentinel `object()` value.
      return True
    elif (isinstance(self.value, DataclassArray) and
          not self.value._array_fields  # pylint: disable=protected-access
         ):
      # Nested dataclass case (if all attributes are `None`, so no active
      # array fields)
      return True
    return False

  @property
  def host_shape(self) -> Shape:
    """Host shape (batch shape shared by all fields)."""
    if not self.inner_shape:
      shape = self.value.shape
    else:
      shape = self.value.shape[:-len(self.inner_shape)]
    # TODO(b/198633198): We need to convert to tuple because TF evaluate
    # empty shapes to True `bool(shape) == True` when `shape=()`
    return tuple(shape)

  def broadcast_to(self, shape: Shape) -> DcOrArrayT:
    """Broadcast the host_shape."""
    final_shape = shape + self.inner_shape
    if self.is_dataclass:
      return self.value.broadcast_to(final_shape)
    else:
      return self.xnp.broadcast_to(self.value, final_shape)


def _validate_dtype(dtype) -> np.dtype:
  """Validate and normalize the dtype."""
  if dtype is int:
    dtype = np.int32
  elif dtype is float:
    dtype = np.float32

  np_dtype = np.dtype(dtype)
  if np_dtype.kind == 'O':
    raise ValueError(f'Array field dtype={dtype} not supported.')
  return np_dtype


def _make_array_field(
    array: DataclassArray,
    field: dataclasses.Field[Any],
    hints: dict[str, TypeAlias],
) -> Optional[_ArrayField]:
  """Make the array field class."""
  # TODO(epot): One possible confusion is if user define
  # `field: Ray = v3d.array_field(shape=(3,))`
  # In which case field will be `float32` instead of `Ray`. Should we raise
  # a warning / error ?
  if _METADATA_KEY in field.metadata:  # Field defined as `= v3d.array_field`:
    field_metadata = field.metadata[_METADATA_KEY]
  # TODO(py3.8):
  # elif field_metadata := _type_to_field_metadata(hints[field.name]):
  else:
    field_metadata = _type_to_field_metadata(hints[field.name])
    if not field_metadata:  # Not an array field
      return None

  return _ArrayField(
      name=field.name,
      host=array,
      **field_metadata.to_dict(),  # pylint: disable=not-a-mapping
  )


def _type_to_field_metadata(hint: TypeAlias) -> Optional[_ArrayFieldMetadata]:
  """Converts type hint to extract `inner_shape`, `dtype`."""
  array_type = type_parsing.get_array_type(hint)
  if isinstance(array_type, type) and issubclass(array_type, DataclassArray):
    # TODO(epot): Should support `ray: Ray[..., 3]` ?
    return _ArrayFieldMetadata(inner_shape=(), dtype=array_type)
  elif isinstance(array_type, array_types.ArrayAliasMeta):
    try:
      return _ArrayFieldMetadata(
          inner_shape=shape_parsing.get_inner_shape(array_type.shape),
          dtype=_validate_dtype(array_type.dtype),
      )
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(e, prefix=f'Invalid shape annotation {hint}.')
  else:  # Not a supported type: Static field
    return None

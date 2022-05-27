# v3d

[![Unittests](https://github.com/google-research/visu3d/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/visu3d/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/visu3d.svg)](https://badge.fury.io/py/visu3d)

### Is v3d for you ?

Yes!

Despite the name, `visu3d` is not limited at all to visualization, nor 3d, but
it can be used in all ML programs (and beyond).

The library provides an abstraction layer on top of TF/Jax/Numpy (same code works
everywhere) at various levels:

*   **For all ML programs:** v3d introduces the `DataclassArray` abstraction which
    significantly reduces boilerplate/verbosity when manipulating datastructures
    by adding numpy-like indexing and vectorization to `dataclasses`. (Future
    plans will move this to an independent module.)

On top of `DataclassArray`, v3d introduces:

*   **For all 3d programs (Nerf, robotics, ...):** standard primitives (camera,
    rays, transformation, ...) that users can use and extend. While those
    primitives can be used as-is in production code, they should also serve as a
    show-off/inspiration of what can be achieved with `DataclassArray`.

Everything in `v3d` is extensible:

*   Your codebase can gradually opt in to specific features you need (e.g.
    trivially migrate your `dataclass` to `v3d.DataclassArray` without any other
    changes).
*   Combine native v3d primitives with your custom ones (see doc below).

On top of the `v3d` primitives:

*   **Best Colab experience:** Everything is trivially visualizable with zero
    boilerplate. Inspect & debug camera poses, trajectories,....

### Core features

<section class="zippy">

Everything is a `v3d.DataclassArray`: **dataclass behave like `np.array`** (with
indexing, slicing, shape manipulation, vectorization,...).

```python
# Single ray
ray = v3d.Ray(pos=[0, 0, 0], dir=[1, 1, 1])
assert rays.shape == ()

# Multiple rays batched together
rays = v3d.Ray(pos=np.zeros((B, H, W, 3)), dir=np.ones((B, H, W, 3)))
assert rays.shape == (B, H, W)

rays = rays.reshape('b h w -> b (h w)')  #  Native `einops` support

top_left_ray = rays[..., 0, 0]  #  (b, h, w) -> (b,)

rays = rays.flatten()
rays = rays[rays.norm() > 0]  # Filter invalid rays
```

</section>
<section class="zippy">

Everything is **visualizable interactively**

Every object has a `.fig` property for interactive visualization:

```python
rays.fig  # Plot the rays
```

Display multiple objects together:

```python
v3d.make_fig([cam, rays, point_cloud])
```

Auto-plot figures with Colab magic:

```python
v3d.auto_plot_figs()  # Once at the start of the Colab

cam, rays, point_cloud  # Tuple auto-displayed without `v3d.make_fig` call
```

</section>
<section class="zippy">

Same code seamlessly **works across Jax, TensorFlow, Numpy** (please help us for
[torch support](https://github.com/google-research/visu3d/issues/12)).

```python
rays = rays.as_jax()  # .as_tf(), as_np(), .as_jax()
assert isinstance(rays.pos, jnp.ndarray)
assert rays.xnp is jnp

rays = rays.as_tf()
assert isinstance(rays.pos, tf.Tensor)
assert rays.xnp is tf.experimental.numpy
```

With native support for auto-diff, `jax.vmap`, `jax.tree_utils`,...

</section>

### Privitives

<section class="zippy">

Common primitives (`Ray`, `Camera`, `Transform`,...), so user can express
intent, instead of math.

```python
H, W = (256, 1024)
cam_spec = v3d.PinholeCamera.from_focal(
    resolution=(H, W),
    focal_in_px=35.,
)
cam = v3d.Camera.from_look_at(
    spec=cam_spec,
    pos=[5, 5, 5],
    target=[0, 0, 0],  # Camera looks at the scene center
)

rays = cam.rays()  # Rays in world coordinates

# Project `(*shape, 3)` world coordinates into `(*shape, 2)` pixel coordinates.
px_coords = cam.px_from_world @ point_cloud
```

See [the API](https://github.com/google-research/visu3d/tree/main/visu3d/__init__.py;l=31)<!-- {.external} !-->
for a full list of primitive.

</section>
<section class="zippy">

Creating your own primitives is trivial.

Converting any dataclass to dataclass array is trivial:

*   Inherit from `v3d.DataclassArray`
*   Use
    [`etils.array_types`](https://github.com/google/etils/blob/main/etils/array_types/README.md)
    to annotate the array fields (or exlicitly use `my_field: Any =
    v3d.array_field(shape=, dtype=)` instead of `dataclasses.field`)

```python
from etils.array_types import FloatArray


@dataclasses.dataclass(frozen=True)
class MyRay(v3d.DataclassArray):
  pos: FloatArray[..., 3]
  dir: FloatArray[..., 3]


rays = MyRay(pos=jnp.zeros((H, W, 3)), dir=jnp.ones((H, W, 3)))
assert rays.shape == (H, W)
```

`v3d` makes it easy to opt-in to the feature you need by implementing the
corresponding protocol.

<!-- See [the tutorial]() for more info. -->

</section>

### Documentation

The best way to get started is to try the Colab tutorials (in the
[docs/](https://github.com/google-research/visu3d/tree/main/docs/)):

*   [Intro (Colab)](https://colab.research.google.com/github/google-research/visu3d/blob/main/docs/intro.ipynb)
    <!-- {.external} !-->
*   [Transform (Colab)](https://colab.research.google.com/github/google-research/visu3d/blob/main/docs/transform.ipynb)
    <!-- {.external} !-->
*   [Create your primitives (Colab)](https://colab.research.google.com/github/google-research/visu3d/blob/main/docs/dataclass.ipynb)
    <!-- {.external} !-->

Installation:

```sh
pip install visu3d
```

Usage:

```python
import visu3d as v3d
```

## Installation

```sh
pip install visu3d
```

*This is not an official Google product.*

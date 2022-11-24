# Changelog

<!--

Changelog follow https://keepachangelog.com/ format.

-->

## [Unreleased]

*   Added: `v3d.make_fig` supports `v3d.make_fig(a, b, c)` (in addition of
    `v3d.make_fig([a, b, c])`)
*   Added: `v3d.fig_config.cam_scale` to globally control the size of the
    cameras.
*   Fixed: Subsampling for displaying simple `np.array` point clouds (regression
    from previous release).

## [1.3.0] - 2022-10-17

### Added

*   `v3d.math` to expose:
    *   Rotation utils (convert from/to rotation matrix)
    *   Subsampling util (`v3d.math.subsample`)
    *   Interpolation util (`v3d.math.interp_points`)
*   `DataclassArray` now supports dynamic shape fields (shape with `None`), like
    `Array[..., None, None]`.
*   More trace customization options:
    *   All `v3d.DataclassParams` objects now have a `.fig_config` property
        (customizable with `obj = obj.replace_fig_config(**options)`).
    *   Automatic subsample customizable with
        `points.replace_fig_config(num_samples=10_000)`.
    *   Object names displayed in plotly are customizable using
        `points.replace_fig_config(name='My point cloud')`.
    *   `v3d.fig_config.num_sample_xyz = 123` to overwrite the default number of
        sampled rays, points,... (`None` for no subsampling)
*   `DataclassArray.__dca_params__` can be set to `v3d.DataclassParams` to
    configure the dataclass options.

### Changed

*   `v3d` dataclass array implementation has been moved to it's independent
    [`dataclass_array`](https://github.com/google-research/dataclass_array)
    package.
*   Any object implementing the `.make_traces` protocol is not visualizable by
    `v3d.make_fig`. No need to inherit `v3d.Visualizable` anymore.
*   Support more complex `DType` (`FloatArray`,... accept `bfloat16`,
    `float64`,...).

### Fixed

*   Translation by subtraction (`ray - np.array([0, 0, 0])`) now works
*   `FigConfig` property (`cam.fig_config`,...) works correctly for batched
    camera.

## [1.2.0] - 2022-05-27

### Changed

*   Camera are now displayed with a complete frame.

## [1.1.0] - 2022-05-13

*   Normalize `look_at` by default

[Unreleased]: https://github.com/google-research/visu3d/compare/v1.3.0...HEAD
[1.2.0]: https://github.com/google-research/visu3d/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/google-research/visu3d/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/google-research/visu3d/releases/tag/v0.3.2

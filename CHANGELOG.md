# Changelog

<!--

Changelog follow https://keepachangelog.com/ format.

-->

## [Unreleased]

### Added

*   `v3d.math` expose rotation utils (convert from/to rotation matrix)
*   Changed: `DataclassArray` now support dynamic shape fields (shape with
    `None`), like `Array[..., None, None]`.
*   Added: `DataclassArray.__dca_params__` can be set to `v3d.DataclassParams`
    to configure the dataclass options.

### Changed

*   Support more complex `DType` (`FloatArray`,... accept `bfloat16`,
    `float64`,...).

## [1.2.0] - 2022-05-27

### Changed

*   Camera are now displayed with a complete frame.

## [1.1.0] - 2022-05-13

*   Normalize `look_at` by default

[Unreleased]: https://github.com/google-research/visu3d/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/google-research/visu3d/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/google-research/visu3d/releases/tag/v0.3.2

# Changelog

All notable changes to this project will be documented in this file.

## Versioning semantic
MAJOR – Breaking changes
MINOR – New features, but backward-compatible
PATCH – Bug fixes, small improvements

## [Unreleased]
### Added
- ...

### Changed
- ...

### Fixed
- ...

## [1.0.1] - 02-07-2025
### Added
- Docstrings with source of material dispersion formulas and input validation for material.py
- Template files abstract interface class for numerical solvers
- Tool for far-field numerical aperture estimation
- Folder for future visualization tools

### Fixed
- `eim_rib` function raising `IndexError: list index out of range` when no slab modes found, now it raises `ValueError` with adquate communicate if `solve_1d_analytic` returns empty list, if not then first element of the returned list is taken for further calculations or is returned

## [1.0.0] - 19-02-2025
### Added
- Initial release with core EIM functionality.
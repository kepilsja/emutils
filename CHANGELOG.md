# Changelog
## [Unreleased] - TYPE
All notable changes to this project will be documented in this file.

## Versioning semantic
MAJOR – Breaking changes

MINOR – New features, but backward-compatible

PATCH – Bug fixes, small improvements

## [1.0.2] - 2025-07-02
### Added
- CI pipeline

## [1.0.1] - 2025-07-02
### Added
- Docstrings with source of material dispersion formulas and input validation for material.py
- Template files abstract interface class for numerical solvers
- Tool for far-field numerical aperture estimation
- Folder for future visualization tools

### Fixed
- `eim_rib` function raising `IndexError: list index out of range` when no slab modes found, now it raises `ValueError` with adquate communicate if `solve_1d_analytic` returns empty list, if not then first element of the returned list is taken for further calculations or is returned

## [1.0.0] - 2025-02-19
### Added
- Initial release with core EIM functionality.
# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4] - 2020-03-09
### Changed
- Better hyperplane tree plotting

### Fixed
- Bug in hyperplane trees (tried to access uninitialized field)

## [0.3] - 2020-02-26
### Added
- Improved scikit-learn compatibility further

### Fixed
- Bug in `model.feature_importance()` computation

## [0.2] - 2019-09-02
### Added
- Experimental support for arbitrarily-oriented hyperplane splits rather than axis-perpendicular ones only
- Experimental support for sparse DataFrames and sparse matrices (scipy.sparse) for fitting and prediction
- Added `model.feature_importance()` for feature selection
- All models now compatible with scikit-learn models

### Changed
- Lots of small changes here and there

## [0.1] - 2019-02-06
### Added
- First release

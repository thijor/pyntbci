# Changelog

## Version 0.2.5

### Added

- Events time-series plot
- (Running) covariance in utilities
- CCA covariance estimator
- CCA cumulative covariance
- On, off and onoff events for rCCA

### Changed
- CCA separate computation for Cxx, Cyy and Cxy
- CCA separate estimators for Cxx and Cyy

### Fixed

- ITR calculation zero-division

## Version 0.2.4

### Added

- CCA cumulative/incremental average and covariance
- Amplitudes (e.g. envelopes) in structure matrix
- Maximum stopping time (`max_time`) for stopping methods
- brainamp64.loc
- A plt.show() in all examples

### Changed

### Fixed

- ITR calculation zero-division

## Version 0.2.3

### Added

### Changed

- Improved documentation
- Improved example pipelines
- Improved tutorial

### Fixed

## Version 0.2.2

### Added

- TRCA transformer
- eTRCA classifier
- Ensemble (`ensemble`) option (i.e., a spatial filter per class) for classifiers

### Changed

- Package name change of PyNT to PyntBCI
- Filterbank order optimized given parameters

### Fixed

- Issue causing novel events in M when "cutting cycles"
- Correlation does not change mutable input variables

## Version 0.2.1

### Added

- Tests
- Tutorial

### Changed

- Non-binary events for rCCA

### Fixed

## Version 0.2.0

### Added

- Dynamic stopping: margin, beta, Bayes
- Inner score metric

### Changed

- All data shapes: trials, channels, samples
- All codes shapes: classes, samples
- Changed all decision functions to similarity, not distance (e.g., Euclidean), to always maximize

### Fixed

- Zero-mean templates in eCCA and rCCA

## Version 0.1.0

### Added

- Filterbank classifier

### Changed

- Classifiers all have predict() and decision_function()

### Fixed

## Version 0.0.2

### Added

### Changed

- CCA method changed from sklearn to covariance method

### Fixed

## Version 0.0.1

### Added

- eCCA template metrics: average, median, OCSVM
- eCCA spatial filter options: all channels or subset

### Changed

### Fixed

## Version 0.0.0

### Added

- CCA transformer
- rCCA classifier
- eCCA classifier

### Changed

### Fixed
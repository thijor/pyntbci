# Changelog

## Version 1.0.0

### Added

- Variable `decoding_length` of `rCCA` in `classifier` controlling the length of a learned spectral filter
- Variable `decoding_stride` of `rCCA` in `classifier` controlling the stride of a learned spectral filter
- Function `decoding_matrix` in `utilities` to phase-shit the EEG data maintaining channel-prime ordering
- Variable `encoding_stride` of `rCCA` in `classifier` controlling the stride of a learned temporal response
- Module `gating` with gating functions, for instance for multi-component or filterbank analysis
- Variable `gating` of `rCCA` in `classifier` to deal with multiple CCA components
- Variable `gating` of `Ensemble` in `classifier`, for example to deal with a filterbank

### Changed

- Variable `codes` of `rCCA` in `classifiers` is renamed to `stimulus`
- Variable `transient_size` of `rCCA` in `classifiers` is renamed to `encoding_length`
- Class `FilterBank` in `classifiers`, is renamed to `Ensemble`
- Function `structure_matrix` in `utilities` is renamed to `encoding_matrix`

### Fixed

- Several documentation issues

## Version 0.2.5 (29-02-2024)

### Added

- Function `eventplot` in `plotting` to plot an event matrix
- Variable `running` of `covariance` in `utilities` to do incremental running covariance updates
- Variable `running` of `CCA` in `transformers` to use a running covariance for CCA 
- Variable `cov_estimator_x` and `cov_estimator_x` of `rCCA` in `classifiers` to change the covariance estimator 
- Event definitions "on", "off" and "onoff" for `event_matrix` in `utilities`

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
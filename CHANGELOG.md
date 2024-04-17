# Changelog

## Version 1.1.0 (17-04-2024)

### Added

- Added `envelope` module containing `envelope_gammatone` and `envelope_rms` functions
- Added `CriterionStopping` to `stopping` for some static stopping methods 

### Changed

- Changed default value of `encoding_length` in `rCCA` of `classifiers` of 0.3 to None, which is equivalent to 1 / fs

### Fixed

- Fixed variable `fs` of type np.ndarray instead of int in examples, tutorials, and pipelines 
- Fixed double call to `decoding_matrix` in `fit` of `rCCA` in `classifiers`

## Version 1.0.1 (26-03-2024)

### Added

- Added `set_stimulus_amplitudes` for `rCCA` in `classifiers`

### Changed

### Fixed

- Fixed dependency between `stimulus` and `amplitudes` in `rCCA` of `classifiers`

## Version 1.0.0 (22-03-2024)

### Added

- Added variable `decoding_length` of `rCCA` in `classifier` controlling the length of a learned spectral filter
- Added variable `decoding_stride` of `rCCA` in `classifier` controlling the stride of a learned spectral filter
- Added function `decoding_matrix` in `utilities` to phase-shit the EEG data maintaining channel-prime ordering
- Added variable `encoding_stride` of `rCCA` in `classifier` controlling the stride of a learned temporal response
- Added module `gating` with gating functions, for instance for multi-component or filterbank analysis
- Added variable `gating` of `rCCA` in `classifier` to deal with multiple CCA components
- Added variable `gating` of `Ensemble` in `classifier`, for example to deal with a filterbank

### Changed

- Changed variable `codes` of `rCCA` in `classifiers` to `stimulus`
- Changed variable `transient_size` of `rCCA` in `classifiers` to `encoding_length`
- Changed class `FilterBank` in `classifiers` to `Ensemble`
- Changed function `structure_matrix` in `utilities` to `encoding_matrix`

### Fixed

- Fixed several documentation issues

## Version 0.2.5 (29-02-2024)

### Added

- Added function `eventplot` in `plotting` to visualize an event matrix
- Added variable `running` of `covariance` in `utilities` to do incremental running covariance updates
- Added variable `running` of `CCA` in `transformers` to use a running covariance for CCA 
- Added variable `cov_estimator_x` and `cov_estimator_m` of `rCCA` in `classifiers` to change the covariance estimator 
- Added event definitions "on", "off" and "onoff" for `event_matrix` in `utilities`

### Changed

- Changed the CCA optimization to contain separate computations for Cxx, Cyy and Cxy
- Changed the CCA to allow separate BaseEstimators for Cxx and Cyy

### Fixed

- Fixed zero-division in `itr` in `utilities` 

## Version 0.2.4

### Added

- Added CCA cumulative/incremental average and covariance
- Added `amplitudes` (e.g. envelopes) in `structure_matrix` of `utilities`
- Added `max_time` to classes in `stopping` to allow a maximum stopping time for stopping methods
- Added brainamp64.loc to capfiles
- Added plt.show() in all examples

### Changed

### Fixed

## Version 0.2.3

### Added

### Changed

- Changed example pipelines to include more examples and explanation
- Changed tutorial pipelines to include more examples and explanation

### Fixed

- Fixed several documentation issues

## Version 0.2.2

### Added

- Added class `TRCA` to `transformers`
- Added class `eTRCA` to `classifiers`
- Added parameter `ensemble` to classes in `classifiers` to allow a separate spatial filter per class

### Changed

- Changed package name from PyNT to PyntBCI to avoid clash with existing pynt library
- Changed filter order in `filterbank` of `utilities` to be optimized given input parameters

### Fixed

- Fixed issue in `rCCA` of `classifiers` causing novel events in structure matrix when "cutting cycles"
- Fixed `correlation` to not contain mutable input variables

## Version 0.2.1

### Added

- Added `tests`
- Added tutorials

### Changed

- Changed `rCCA` to work with non-binary events instead of binary only

### Fixed

## Version 0.2.0

### Added

- Added dynamic stopping: classes `MarginStopping`, `BetaStopping`, and `BayesStopping` in module `stopping`
- Added value inner for variable `score_metric` in 'classifiers'

### Changed

- Changed all data shapes from (channels, samples, trials) to (trials, channels, samples)
- Changed all codes shapes from (samples, classes) to (classes, samples)
- Changed all decision functions to similarity, not distance (e.g., Euclidean), to always maximize

### Fixed

- Fixed zero-mean templates in `eCCA` and `rCCA` of `classifiers`

## Version 0.1.0

### Added

- Added `Filterbank` to `classifiers`

### Changed

- Changed classifiers all have `predict` and `decision_function` methods in `classifiers`

### Fixed

## Version 0.0.2

### Added

### Changed

- Changed CCA method from sklearn to custom covariance method

### Fixed

## Version 0.0.1

### Added

- Added `eCCA` template metrics: average, median, OCSVM
- Added `eCCA` spatial filter options: all channels or subset

### Changed

### Fixed

## Version 0.0.0

### Added

- Added `CCA` in `transformers`
- Added `rCCA` in `classifiers`
- Added `eCCA` in `classifier`

### Changed

### Fixed

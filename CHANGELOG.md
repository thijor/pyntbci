# Changelog

## Version 1.8.3 (21-05-2025)

### Added
- Added `approach` in `BayesStopping` of `stopping`

### Changed
- Refactor `rCCA` of `classifiers`

### Fixed
- Fixed `astype` in all modules
- Fixed `decoding_matrix` in `rCCA` of `classifiers` only called if required
- Fixed `encoding_stride` in `rCCA` of `classifiers` to allow list input like `encoding_length`
- Fixed docs

## Version 1.8.2 (16-04-2025)

### Added

### Changed
- Changed `stride` in `encoding_matrix` of `utilities` to allow list input like `length`

### Fixed
- Fixed `cca_channels` in `eCCA` of `classifiers`
- Fixed `lags=[...]` and `ensemble=True` combination in `eCCA` of `classifiers`

## Version 1.8.1 (11-03-2025)

### Added
- Added `labels` to `stimplot` in `plotting`

### Changed
- Changed `pinv` in `utilities` to work with non-square matrices

### Fixed
- Fixed array `encoding_length` of `rCCA` in `classifiers`
- Fixed `smooth_width` of `CriterionStopping` in `stopping`
- Fixed `stop_time_` of `CriterionStopping` in `stopping`
- Fixed `gamma_x` and `gamma_y` regularization of `CCA` in `transformers`

## Version 1.8.0 (08-11-2024)

### Added
- Added `min_time` to stopping methods in `stopping`
- Added `max_time` to `CriterionStopping` in `stopping`

### Changed

### Fixed
- Fixed fit exception in `DistributionStopping` in `stopping`

## Version 1.7.0 (22-10-2024)

### Added
- Added `tmin` to `encoding_matrix` in `utilities`
- Added `tmin` to `rCCA` in `classifiers`

### Changed

### Fixed

## Version 1.6.1 (10-10-2024)

### Added
- Added `find_neighbours` and `find_worst_neighbour` to `utilities`
- Added `optimize_subset_clustering` to `stimulus`
- Added `optimize_layout_incremental` to `stimulus`
- Added `stimplot` to `plotting`

### Changed
- Changed order of tutorials and examples

### Fixed
- Fixed `max_time` in all `stopping` classes to deal with "partial" segments

## Version 1.5.0 (30-09-2024)

### Added
- Added `ValueStopping` to `stopping`
- Added parameter `distribution` to `DistributionStopping` in `stopping` 

### Changed
- Changed `envelope_rms` to `rms` in `envelope`
- Changed `envelope_gammatone` to `gammatone` in `envelope`
- Changed `BetaStopping` in `stopping` to `DistributionStopping`

### Fixed
- Fixed default `CCA` in `transformers` to `inv`, not `pinv`
- Fixed `seed` for `make_m_sequence` and `make_gold_codes` in `stimulus` to not be full zeros

## Version 1.4.1 (19-07-2024)

### Added

### Changed

### Fixed
- Fixed default `CCA` in `transformers` to `inv`, not `pinv`

## Version 1.4.0 (15-07-2024)

### Added
- Added `pinv` to `utilities`
- Added `alpha_x` to `CCA` in `tranformers`
- Added `alpha_y` to `CCA` in `tranformers`
- Added `alpha_x` to `eCCA` in `classifiers`
- Added `alpha_t` to `eCCA` in `classifiers`
- Added `alpha_x` to `rCCA` in `classifiers`
- Added `alpha_m` to `rCCA` in `classifiers`
- Added `squeeze_components` to `rCCA`, `eCCA`, `eTRCA` in `classifiers'

### Changed
- Changed `numpy` typing of `np.ndarray` to `NDArray`
- Changed `cca_` and `trca_` attributes to be `list` always in `eCCA`, `rCCA` and `eTRCA`
- Changed `scipy.linalg.inv` to `pyntbci.utilities.pinv` in `CCA` of `transformers`
- Changed `decision_function` and `predict` of `classifiers` to return without additional dimension for components if `n_components=1` and `squeeze_components=True`, both of which are defaults

### Fixed

## Version 1.3.3 (01-07-2024)

### Added

### Changed

### Fixed
- Fixed components bug in `decision_function` of `eCCA` in `classifiers` 

## Version 1.3.2 (23-06-2024)

### Added
- Added `cov_estimator_t` to `eCCA` in `classifiers`

### Changed
- Changed separate covariance estimators for data and templates in `eCCA` of `classifiers`

### Fixed

## Version 1.3.1 (23-06-2024)

### Added

### Changed

### Fixed
- Fixed zero division `eventplot` in `plotting`
- Fixed event order duration event `event_matrix` in `utilities` 

## Version 1.3.0 (18-06-2024)

### Added
- Removed `gating` of `rCCA` in `classifiers`
- Removed `_score` methods in `classifiers`
- Added `n_components` in `eCCA` in `classifiers`
- Added `n_components` in `eTRCA` in `classifiers`

### Changed
- Changed "bes" to "bds" in `BayesStopping` in `stopping` in line with publication
- Changed `lx` and `ly` to `gamma_x` and `gamma_y` iof `eCCA` in `classifiers`
- Changed `gating` to `gates`
- Changed `TRCA` in `transformers` to deal with one-class data only
- Changed `_get_T` to `get_T` in all `classifiers`

### Fixed

## Version 1.2.0 (18-04-2024)

### Added

### Changed

- Changed `lx` of `rCCA` in `classifiers` to `gamma_x`, which ranges between 0-1, such that the parameter represents shrinkage regularization
- Changed `ly` of `rCCA` in `classifiers` to `gamma_m`, which ranges between 0-1, such that the parameter represents shrinkage regularization
- Changed `lx` of `CCA` in `transformers` to `gamma_x`, which ranges between 0-1, such that the parameter represents shrinkage regularization
- Changed `ly` of `CCA` in `transformers` to `gamma_y`, which ranges between 0-1, such that the parameter represents shrinkage regularization

### Fixed

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

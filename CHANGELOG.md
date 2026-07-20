# Changelog

## Version 1.9.0

### Added
- Added `vmin` and `vmax` to `topoplot` in `plotting`
- Added `diff` event to `event_matrix` in `utilities`
- Added `eeg` module to generate synthetic multi-channel EEG data (c-VEP signal, and noise) 
- Added `Vectorizer` to `transformers`
- Added `classes_` attribute to `eCCA`, `Ensemble`, `rCCA` in `classifiers`, `AggregateGate`, `DifferenceGate` in
  `gates`, and all classes in `stopping`, for scikit-learn `ClassifierMixin` compatibility
- Added a clear, actionable error message in `CCA` of `transformers` when a covariance matrix is singular or too
  ill-conditioned to invert (e.g. `ensemble=True` in `eCCA`/`rCCA` with too little data per class), instead of a bare
  "singular matrix" error or, worse, a silently corrupted result; documented this risk on `ensemble` in `eCCA`/`rCCA`
- Added a `dtype` parameter (default: `"float32"`) to every generator function in `eeg`; all computation is still
  done internally in float64 (float64/complex128 for the FFT in `generate_pink_noise`) for numerical precision, and
  only cast to the requested `dtype` on the returned array
- Added `inner` and a `running` mode (with a `*_old`/`n_old`-style state, matching `covariance`'s existing convention)
  to `correlation` and `euclidean` in `utilities`, letting a score matrix be updated with only newly observed samples
  instead of recomputed from scratch; `correlation`'s running mode is implemented by reusing `covariance`'s existing
  running mechanism directly (correlation is covariance normalized by the two variances, verified numerically
  identical), while `euclidean`/`inner` accumulate their own (simpler, since they need no running mean) raw sums
- Added `running`/`reset` parameters to `decision_function`/`predict` in `eCCA`/`rCCA` of `classifiers` (not
  `ensemble=True`), letting them be fed only the newly observed samples of a growing trial instead of recomputing
  the full spatial (and, for `rCCA`, spatio-spectral) filter and score from scratch every call; for `rCCA`,
  `decoding_matrix`'s forward-looking window is handled by keeping a small raw-sample buffer and only committing a
  position's contribution once no future sample can still change it. All 5 `stopping` classes now use this
  internally in `fit()`'s calibration loop, and expose the same `running`/`reset` parameters on their own
  `predict()`; for a wrapped `estimator` that isn't an `eCCA`/`rCCA` (or has `ensemble=True`), a transparent
  fallback buffers the raw data and recomputes from scratch instead, so `running=True` still works (just without
  the speedup). Verified numerically identical to the non-running computation (max abs error ~1e-13, floating-point
  noise) across every score metric, `n_components`, and `rCCA`'s `decoding_length`/`decoding_stride` combinations;
  measured 21x faster `decision_function` calls and 2x faster `fit()` calibration on an 8.4s trial (84 segments).
  This targets the *predict-time* per-trial scoring loop specifically; `CCA` in `transformers` already has a
  separate, pre-existing `running` mode for *fit-time* incremental covariance estimation across successive `fit()`
  calls (used to train the spatial filter itself) — the two are unrelated to each other by design, since predict-time
  running state (an in-progress trial's scores) and fit-time running state (the model's learned covariance) have
  different lifecycles, but both are built on the same underlying `covariance()` primitive
- Added `examples/example_6_moabb.py`, the first example to run on real (rather than synthetic) EEG data: the
  Thielen (2015) c-VEP dataset, loaded via MOABB and cross-validated with plain scikit-learn (`StratifiedKFold` +
  `cross_val_score`), since `rCCA` is itself a scikit-learn compatible estimator; needs the optional `moabb`/`mne`
  dependencies and downloads ~450 MB of data on first use, so it is excluded from execution when building the
  documentation (see `filename_pattern` in `doc/conf.py`)
- Added test coverage for `Ensemble` in `classifiers` (previously untested), the entire `envelope` module (previously
  had no test file), `eventplot`/`stimplot` in `plotting` (previously only `topoplot` was tested), `find_neighbours`,
  `find_worst_neighbour`, `pinv`, and `trials_to_epochs` in `utilities` (previously untested), and a new
  `test_sklearn_compliance.py` that checks `get_params()`/`set_params()`/`clone()` round-tripping across every
  classifier, gate, and stopping estimator in the library
- Added classification-correctness tests (`accuracy = mean(yh == y)` against a threshold, not just output shape) for
  `eCCA`, `rCCA`, `Ensemble` in `classifiers`, `AggregateGate`, `DifferenceGate` in `gates`, and all classes in
  `stopping`; previously every one of these tests only checked output shape, so a classifier that always predicted
  a constant class, or a `decision_function` computed backwards, would have still passed the whole suite
- Added a `running` parameter to `eCCA`/`rCCA` in `classifiers`, letting `fit()` be called repeatedly with new
  batches of trials that *add to* the previous fit instead of replacing it, by reusing `CCA`'s own pre-existing
  `running=True` mechanism for the spatial (and, for `rCCA`, spatio-temporal) filter's covariance. For `rCCA` this
  is mathematically exact: its templates (`Ts_`/`Tw_`) are derived purely from the (fixed) stimulus and the current
  filter, never from the training trials, so `fit(X1, y1)` then `fit(X2, y2)` gives the identical filter (verified
  to ~1e-8) as one `fit(concat(X1, X2), concat(y1, y2))` call. For `eCCA` it is necessarily an approximation, since
  its template is itself an average of the training trials and is used as the CCA fit's target on every call, so
  earlier calls see a less complete estimate of it than later ones; it converges towards the batch result as trials
  accumulate (verified: >0.999 cosine similarity to the batch filter after a few batches) but is not expected to
  equal it exactly, and is scoped to `lags` being set, `template_metric="mean"`, and `ensemble=False` (the only
  combination with both a fixed, known-upfront class count and a single running template, rather than one needing
  to grow dynamically as new classes are observed or with no exact incremental form). Both raise a clear error on a
  channel/sample-count mismatch between calls, and both correctly restart a fresh running sequence (rather than
  silently resuming a stale one) if `running` is toggled off and back on between `fit()` calls. This targets
  *fit-time* incremental training, the counterpart to the *predict-time* running scoring added above; the two are
  unrelated by design but share the same `covariance()` primitive underneath

### Changed
- Removed the bundled example/tutorial EEG data (`data/`) from the package; `tutorials/` and `examples/` now generate
  synthetic data with `eeg` instead
- Removed the `pipelines/` folder of standalone analysis scripts for external published datasets
- Removed the `mne` dependency; `examples/` now use `Vectorizer` in `transformers` instead of `mne.decoding.Vectorizer`
- Removed the 'seaborn' dependency; now simply relies on matplotlib
- Removed `eTRCA` from `classifiers` and `TRCA` from `transformers`
- Removed the `estimator` parameter from `covariance` in `utilities` (and, with it, `estimator_x`/`estimator_y` from
  `CCA` in `transformers` and `cov_estimator_x`/`cov_estimator_t`/`cov_estimator_m` from `eCCA`/`rCCA` in
  `classifiers`), along with the `NotImplementedError` it raised whenever a custom estimator was combined with
  `running=True` past the first call; the empirical covariance formula (the only implemented, tested path, since
  the custom-estimator path had no running-mode implementation to begin with) is used unconditionally instead
- Dropped Python 3.8 support (minimum is now 3.9), since the codebase already relied on builtin generic type hints
  (e.g. `tuple[...]`) that are not subscriptable at runtime on 3.8
- Migrated packaging metadata from the legacy `setup.cfg` to a PEP 621 `[project]` table in `pyproject.toml`
  (`version` and `readme` remain dynamic, sourced from `pyntbci.__version__` and `README.md`/`CHANGELOG.md`
  respectively, as before); verified the built sdist/wheel metadata and bundled files are unchanged

### Fixed
- Fixed `BayesStopping` in `stopping` check fitted if score-based
- Fixed `itr` in `utilities` output shape is input shape
- Fixed retaining dtype in `stimulus`
- Fixed `DistributionStopping` in `stopping` not checking fitted status when `trained=False`
- Fixed `AggregateGate` in `gates` modifying its `aggregate` parameter in `__init__`, which broke scikit-learn
  `clone()`
- Fixed `rCCA` in `classifiers` doing parameter resolution and computation in `__init__` instead of `fit`, which left
  the estimator in a stale, inconsistent state after `set_params()`
- Fixed `DistributionStopping` in `stopping` validating `distribution` eagerly in `__init__` instead of `fit`
- Fixed `correlation` in `utilities`, `encoding_matrix` in `utilities` (`stimulus` and length-1 `length`/`stride`
  lists), `itr` in `utilities`, `get_T` in `eCCA` of `classifiers`, and `transform` in `CCA` of `transformers` all
  mutating their input arguments in place as a side effect
- Fixed `gamma_x`/`gamma_y` regularization in `CCA` of `transformers` silently computing wrong (asymmetric, not
  positive semi-definite) covariance matrices whenever a per-feature array (rather than a single scalar) was used;
  scalar `gamma_x`/`gamma_y` behavior is unchanged
- Fixed `examples/` and tests pre-computing an unseeded `y` externally and passing it into `generate_c_vep` of `eeg`,
  which bypassed `eeg`'s own seeded label assignment and made `random_state` not actually reproduce the data;
  `generate_c_vep` itself was already correctly reproducible when `y` is left to be generated internally (i.e. by
  passing `n_classes` instead of a pre-computed `y`)
- Fixed `gammatone` in `envelope` not applying its lowpass filter to the envelope at all (`filtfilt` was called with
  the denominator coefficients replaced by `1`, i.e. an FIR-only pass), and switched that filter to second-order
  sections (`sosfiltfilt`) since the transfer-function (`b`/`a`) form of the actual (high-order) filter is
  numerically unstable and produced `NaN` once the filter was applied at all
- Fixed `DistributionStopping` in `stopping` swapping the wrong axis when moving the target class's score to index 0
  before fitting the non-target score distribution (`trained=True`), which corrupted every fitted distribution and
  could raise an `IndexError`
- Fixed `BayesStopping` in `stopping` mutating its own `target_pf`/`target_pd` parameters as a side effect of
  `predict()` (methods `bds1`/`bds2`), which meant repeated `predict()` calls could return different results and
  `clone()`/`get_params()` no longer reflected the original estimator after a prediction; the (per-prediction) target
  adjustment is now local instead, and reported with `warnings.warn` instead of `print`
- Fixed `BayesStopping` in `stopping` (`method="bds2"`) raising an unguarded `IndexError` when no segment jointly
  satisfies both the `target_pf` and `target_pd` constraints; now falls back to the most conservative (last) segment
- Fixed `CriterionStopping` in `stopping` silently producing `nan` scores (and thus a meaningless `stop_time_`) when
  `n_trials < n_folds` left some cross-validation folds with no test data; now raises an assertion instead
- Fixed `topoplot` in `plotting` raising a `ValueError` when reading a `.loc` file with a trailing newline
- Fixed `optimize_layout_incremental` in `stimulus` using the unseeded global `np.random.permutation` for its random
  initial layouts, unlike every other randomized function in the library; added a `random_state` parameter
- Fixed minor internal consistency issues: `AggregateGate`/`DifferenceGate` in `gates` now use the same
  `ClassifierMixin, BaseEstimator` MRO order as every other estimator in the library; `CriterionStopping.predict` in
  `stopping` now returns `int64` (not `float64`) for unstopped (`-1`) trials, matching every other stopping class;
  `DistributionStopping` in `stopping` now implements `__sklearn_is_fitted__` instead of checking a private attribute
  directly through `check_is_fitted`, matching `Ensemble` in `classifiers`
- Fixed `stimplot`'s `upsample` docstring in `plotting`, copy-pasted from `eventplot` and referring to a
  non-existent "event time-series"
- Fixed `eventplot` in `plotting` not validating that `events` (if provided) has one name per row of `E`, unlike the
  equivalent `labels` check in `stimplot`
- Fixed the `documentation` CI workflow committing and pushing to `gh-pages` on pull requests as well as pushes to
  `main`; the build now only deploys on `push`, and still runs as a build-only check on pull requests
- Fixed `test_correlation_faster_corrcoef` in the test suite being a wall-clock timing comparison that could fail
  under CI load for reasons unrelated to the implementation; it is now skipped by default and kept for manual
  benchmarking only
- Fixed `pinv` in `utilities` computing `U @ diag(1/d) @ Vh` instead of the true Moore-Penrose pseudo-inverse
  `Vh.T @ diag(1/d) @ U.T`; both are equivalent (and both were already correct) for the symmetric covariance
  matrices `pinv` is used on internally in `CCA` of `transformers`, but the old formula silently returned a
  wrong-shaped, mathematically incorrect result for any general (non-symmetric or non-square) matrix
- Improved performance of `euclidean` in `utilities` (vectorized, was an O(n_A * n_B) Python loop), `decoding_matrix`
  and `encoding_matrix` in `utilities` (avoid computing and discarding the wrapped-around part of `np.roll` on every
  window), the inverse square root in `CCA` of `transformers` (uses the symmetric eigendecomposition instead of
  the general-purpose `scipy.linalg.sqrtm`, which also removes the need to discard spurious imaginary numerical
  noise with `np.real`), `is_gold_code` in `stimulus` (replaced the O(n_classes^2 * n_bits) loop of `np.roll`
  calls with a single vectorized correlation over all circular shifts, gathered by indexing into two concatenated
  code cycles), `transform` in `CCA` of `transformers` for 3D input (batched matmul directly on the 3D array
  instead of a transpose+reshape that forced a full copy of the data on every call), `_compute_difference_scores` in
  `DifferenceGate` of `gates` (replaced the Python double loop over class pairs with a single `np.triu_indices`
  gather), and `topoplot` in `plotting` (replaced the Python double loop over the 300x300 interpolation grid that
  masked points outside the head radius with a single vectorized broadcast comparison)
- Fixed `decision_function`/`predict` in `eCCA`/`rCCA` of `classifiers` silently returning a meaningless, always-the-
  same-class prediction (`argmax` over a NaN or all-equal-score row) instead of raising, when `running=True` was
  called with a zero-sample chunk on the very first call of a sequence (before any real data had been observed);
  now asserts at least 1 sample is available before the first score can be computed. A zero-sample chunk *after*
  real data has already been observed remains a well-defined no-op, as intended

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

# Contributing to PyntBCI

Thank you for considering contributing to PyntBCI!

## Development setup

Clone the repository and install it in editable mode along with the test and documentation dependencies:

	git clone https://github.com/thijor/pyntbci.git
	cd pyntbci
	pip install -e .
	pip install pytest ruff

## Running tests

Tests live in `pyntbci/tests/` and are written with `unittest`, discoverable by `pytest`:

	pytest pyntbci/tests

Please add or update tests for any change in behavior. All tests must pass before a pull request can be merged; this is enforced by CI (`.github/workflows/test.yml`) on Python 3.8 through 3.12.

## Linting and formatting

Code style is enforced with [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`:

	ruff check .
	ruff format .

Both `ruff check .` and `ruff format --check .` run in CI and must pass before a pull request can be merged.

## Building the documentation

The documentation is built with Sphinx:

	pip install sphinx sphinx_rtd_theme myst_parser sphinx-gallery
	cd doc/
	make html

The built HTML will be in `doc/_build/html/`. Documentation is automatically built and deployed to GitHub Pages on every push to `main` (`.github/workflows/documentation.yml`).

## Submitting changes

1. Fork the repository and create a branch for your change.
2. Make your change, including tests and documentation updates where relevant.
3. Update `CHANGELOG.md` under an "Unreleased" or the current in-progress version heading.
4. Open a pull request describing the motivation and the change.

## Reporting issues

Please use the GitHub issue tracker at https://github.com/thijor/pyntbci/issues to report bugs or request features.

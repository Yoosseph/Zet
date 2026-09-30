# Contributing to Zet

Thanks for helping improve Zet. It is an early-stage project; small bug fixes, documentation
corrections, and reproducible test cases are especially useful.

## Development setup

Use Python 3.10 or newer. From the repository root:

```sh
python -m venv .venv
python -m pip install -e ".[test]"
python -m pytest
```

Activate the virtual environment or use its Python executable for the last two commands. The
ordinary test suite uses tiny committed ONNX fixtures and does not download model weights. Real
model tests require `--run-model` and a local checkpoint; see [the run guide](docs/HOW-TO-RUN.md).

Before sending a pull request, run the tests relevant to your change. Explain what changed, why,
and how you tested it. For changes to calibration or reported numbers, include the data source,
sample size, seed, and model revision. Record non-obvious design choices in
[docs/decisions.md](docs/decisions.md).

## Protect user data

Do not commit real customer messages, task directories, credentials, model caches, or local
machine reports. Use synthetic examples with invented details in tests and issues. Review every
staged file with `git diff --cached` before committing; `.gitignore` does not hide files that Git
already tracks. See [docs/PRIVACY.md](docs/PRIVACY.md).

Code you contribute is accepted under this repository's Apache-2.0 license. If you copy code or
data from elsewhere, identify its source and license and preserve required notices. Zet includes
modified Laya code, with attribution in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

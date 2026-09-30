# Releasing Zet

Zet is currently `0.1.0.dev0`: the source is public, but there is no versioned GitHub or PyPI
release yet. The benchmark describes what this development version has actually measured.

## Before cutting a version

1. Decide the release version and update `zet/__init__.py`, the README status line, and
   `CHANGELOG.md`. `pyproject.toml` reads the version from `zet/__init__.py`.
2. Review [benchmark.md](benchmark.md), its [raw results](benchmarks/massive/results.json), and
   the known limits. Do not describe the benchmark as proof of performance on customer emails.
3. Run `python -m pytest -q` and the real-model checks described in
   [docs/HOW-TO-RUN.md](docs/HOW-TO-RUN.md). Confirm the GitHub CI test and package jobs pass for
   the exact commit to release.
4. Build with `python -m pip install build` and `python -m build`. Inspect the source archive and
   wheel. The source archive should contain the license, notices, benchmark, results, and release
   guide; the wheel should contain `zet/ui/static/index.html`.
5. Review `git status`, `git diff --cached`, and the files in the archives for private data.
   Follow [docs/PRIVACY.md](docs/PRIVACY.md). The model weights and MASSIVE source dataset are
   downloaded separately and should not be bundled.
6. Once the version and artifacts are approved, tag the release, publish GitHub release notes,
   and only then publish to a package index if the package name and publishing credentials are
   ready. Keep the GitHub and package versions identical.

The six-topic MASSIVE result shows useful routing on that dataset. The 18-topic benchmark and the
synthetic support-email fixture did not automate answers at a 5% budget. A claim about usefulness
on real email workflows needs representative labeled emails and a trial with actual reviewers.

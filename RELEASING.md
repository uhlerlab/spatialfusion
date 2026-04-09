# Releasing SpatialFusion to PyPI
TL;DR Release checklist

1. Bump version in `pyproject.toml`
2. Merge to `main`
3. Tag release commit
4. Build artifacts
5. Upload to TestPyPI
6. Validate install
7. Upload to PyPI
8. Create GitHub release

This document describes the process for publishing a new SpatialFusion
release to PyPI and creating the corresponding GitHub release.

## 0) One-time setup

Create accounts and API tokens:
- TestPyPI: https://test.pypi.org/account/register/
- PyPI: https://pypi.org/account/register/

Create API tokens:
- TestPyPI token: https://test.pypi.org/manage/account/token/
- PyPI token: https://pypi.org/manage/account/token/

Create a `.pypirc` file (`~/.pypirc`):
```
[pypi]
username = __token__
password = pypi-xxxxx

[testpypi]
username = __token__
password = pypi-yyyyy
```
Install release tools in your local environment:

```bash
python -m pip install -U build twine
```

## 1) Finalize code changes

1. Make all desired changes.
2. Ensure tests/checks you care about pass.
3. Bump package version manually in `pyproject.toml`. For example, `0.0.9` -> `0.1.0`
4. Commit and push your branch.
5. Merge to `main`.

## 2) Create and push a git tag

Create an annotated tag on the release commit:

```bash
git checkout main
git pull
git tag -a v0.1.0 -m "SpatialFusion v0.1.0"
git push origin v0.1.0
```

## 3) Build artifacts

Clean old artifacts and build fresh:

```bash
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
ls -lh dist/
```

## 4) Upload to TestPyPI
TestPyPI allows validating the package upload and installation process
without publishing a permanent release to the real PyPI registry.


```bash
python -m twine upload --repository testpypi dist/* --verbose
```

## 5) Install from TestPyPI and validate

Create a clean test environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Follow README.md installation instructions, but installing spatialfusion from TestPyPI:

```bash
# install cpu-based requirements
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
# install spatialfusion
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  "spatialfusion==0.1.0"
```

Then run the tutorial notebook workflow (for example post-unimodal steps) to verify behavior.

## 6) Upload to real PyPI

```bash
python -m twine upload dist/* --verbose
```

## 7) Install from real PyPI and validate

Create another clean environment and install from PyPI:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# install cpu-based requirements
pip install "torch==2.4.1" "torchvision==0.19.1" \
  --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
# install spatialfusion
pip install "spatialfusion==0.1.0"
```

Run the same tutorial check used for TestPyPI.

## 8) Create GitHub release

1. Open the GitHub repository releases page.
2. Create a new release from the pushed tag (for example `v0.1.0`).
3. Add release notes/changelog highlights.
4. Publish the release.

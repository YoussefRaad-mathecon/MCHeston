# How to publish MCHeston

Everything in this guide is **pre-filled with your specific values** —
your name, email, GitHub username, and package name are already in the
repo. You just need to follow the steps.

## Before you start

The package name on PyPI is **MCHeston** (case-insensitive, so
`mcheston` and `MCHeston` are the same project).

The Python import name is **`mcheston`** (lowercase). Users will type:

```bash
pip install MCHeston           # install
```
```python
from mcheston import HestonModel   # import in code
```

## Step 1: Check the name is available

Open https://pypi.org/project/mcheston/ in a browser.

* If it says **"404 Not Found"** → the name is free, you're good.
* If it shows an existing project → someone has it. You'll need to
  pick a different name (e.g. `mcheston-yraad`) and edit
  `pyproject.toml` line 5: `name = "MCHeston"` → `name = "mcheston-yraad"`.

## Step 2: Push to GitHub

You already have https://github.com/YoussefRaad-mathecon. Create a
new repo there (the green "New" button at the top right) named
**MCHeston**. Don't tick "add README" or "add .gitignore" — we already
have those.

Then from the unzipped folder on your machine:

```bash
cd MCHeston
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YoussefRaad-mathecon/MCHeston.git
git push -u origin main
```

After this push, GitHub Actions will automatically run your tests. You
can watch them at:
https://github.com/YoussefRaad-mathecon/MCHeston/actions

The first run takes ~3 minutes (it tests on Linux, macOS, Windows × four
Python versions = 12 combinations).

## Step 3: Make a PyPI account

1. Register on **Test PyPI first** (sandbox for practice):
   https://test.pypi.org/account/register/
2. Then on the **real PyPI** (final destination):
   https://pypi.org/account/register/

Note: these are two completely separate accounts/passwords. They don't
share anything.

For each account, after registering:

1. Verify your email (they send a link).
2. Go to *Account settings → API tokens → Add API token*.
3. Token name: anything (e.g. `mcheston-upload`).
4. Scope: "Entire account" (you can scope it down to MCHeston later).
5. Click "Create token". **Copy the token immediately** — it's shown
   exactly once. The token starts with `pypi-`.

You'll have **two tokens** — one for Test PyPI, one for real PyPI.
Keep them somewhere safe.

## Step 4: Build the distribution

From the repo root:

```bash
pip install --upgrade build twine
python -m build
```

This creates a `dist/` folder with two files:

```
dist/MCHeston-0.1.0-py3-none-any.whl     <- the wheel
dist/MCHeston-0.1.0.tar.gz                <- the source tarball
```

Validate them:

```bash
python -m twine check dist/*
```

Both should say **PASSED**. If not, stop and fix whatever it complains
about — PyPI will reject the upload otherwise.

## Step 5: Upload to Test PyPI first

Strongly recommended before the real upload:

```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:
* **Username:** `__token__` (literally those eight characters: two
  underscores + "token" + two underscores)
* **Password:** paste your **Test PyPI** token (starts with `pypi-`)

If it succeeds, install from Test PyPI in a fresh environment to verify:

```bash
# Linux / macOS:
python -m venv /tmp/test-mch
source /tmp/test-mch/bin/activate

# Windows PowerShell:
# python -m venv $env:TEMP\test-mch
# $env:TEMP\test-mch\Scripts\Activate.ps1

pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ MCHeston

python -c "from mcheston import HestonModel; print('it works')"

deactivate
```

(The `--extra-index-url` is needed because numpy/scipy live on the real
PyPI, not Test PyPI.)

## Step 6: Upload to real PyPI

Once Test PyPI worked:

```bash
python -m twine upload dist/*
```

* **Username:** `__token__`
* **Password:** paste your **real PyPI** token

People can now install your package with:

```bash
pip install MCHeston
```

It will appear at https://pypi.org/project/MCHeston/.

## Step 7: Releasing future versions

Once you start adding features (puts, Greeks, calibration, etc.):

1. Bump the version in two places:
   * `pyproject.toml` → `version = "0.1.1"`
   * `src/mcheston/__init__.py` → `__version__ = "0.1.1"`
2. Commit and tag:
   ```bash
   git commit -am "Release 0.1.1"
   git tag v0.1.1
   git push --tags
   ```
3. Clean and rebuild:
   ```bash
   rm -rf dist/ build/ src/*.egg-info
   python -m build
   python -m twine upload dist/*
   ```

PyPI does not allow re-uploading the same version, so always bump first.

## Common gotchas

* **"File already exists" on upload** — you tried to re-upload the same
  version. Bump the version number in both places (pyproject.toml +
  __init__.py) and rebuild.
* **"Name already taken"** — someone else took MCHeston between
  Step 1 and Step 6. Edit `name = "..."` in `pyproject.toml`.
* **`pip install` finds an old version** — clear pip's cache:
  `pip cache purge`, or use `pip install --no-cache-dir MCHeston`.
* **CI badge in README shows "no status"** — wait until the first
  workflow run completes; the badge updates automatically afterward.
* **Tests pass locally but fail in CI** — usually a path/import issue.
  Check the GitHub Actions log for the exact error.

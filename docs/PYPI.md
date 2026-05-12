# Publishing PGMST to PyPI

This project uses **[PEP 621](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/)** metadata in `pyproject.toml` and **[hatchling](https://hatch.pypa.io/latest/config/build/)** as the build backend.

## 1. One-time setup

```bash
python -m pip install --upgrade pip build twine hatch
```

Register accounts on [PyPI](https://pypi.org/account/register/) and [TestPyPI](https://test.pypi.org/account/register/) if needed.

**Option A — GitHub Actions (recommended):** configure a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for the `pgmst` project on PyPI:

1. PyPI → your project (create `pgmst` if it does not exist) → **Manage** → **Publishing** → **Add a new pending publisher**.
2. **PyPI Project name:** `pgmst`  
   **Owner:** `UrbanGISer`  
   **Repository name:** `PGMST`  
   **Workflow name:** `publish-pypi.yml`  
   **Environment name:** leave **blank** (workflow does not use a named environment), unless you add `environment:` in the workflow and match it here.
3. Save, then merge the workflow file from this repo and create a **GitHub Release** (or push a matching tag if you adjust the trigger). The workflow uploads the built artifacts to PyPI using OIDC (no long-lived PyPI token in GitHub secrets).

**Option B — Manual upload:** configure API tokens or `~/.pypirc` and use `twine upload` as below.

## 2. Update metadata before release

- Bump **`version`** in `pyproject.toml` (and keep `pgmst/__init__.py` `__version__` in sync).
- Confirm **`[project.urls]`** in `pyproject.toml` matches the GitHub repo.
- Refresh **`README.md`** changelog or highlights if you like.

## 3. Build artifacts

From the **repository root** (the folder that contains `pyproject.toml`):

```bash
python -m build --sdist --wheel
```

This creates `dist/pgmst-<version>.tar.gz` and `dist/pgmst-<version>-py3-none-any.whl` (or similar).

## 4. Check the wheel

```bash
twine check dist/*
```

## 5. Upload to TestPyPI (recommended first)

```bash
twine upload --repository testpypi dist/*
```

Test install:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pgmst
```

## 6. Upload to PyPI

```bash
twine upload dist/*
```

## 7. Notes

- **Name collision**: if `pgmst` is taken on PyPI, change `project.name` in `pyproject.toml` (e.g. `pgmst-zoning`) and update `import` examples in the README.
- **Optional extras**: `pip install "pgmst[dev]"` installs `twine`, `build`, `hatch` for maintainers.
- The **Florida demo** notebook is included in the wheel under `pgmst/examples/` via `force-include` in `pyproject.toml`.

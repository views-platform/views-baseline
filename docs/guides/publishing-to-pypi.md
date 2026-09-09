# Publishing `views-baseline` to PyPI

A practical runbook for releasing this package, mirroring the `views-frames` /
`views-datafactory` / `views-reporting` routine. Written to be followed **solo, cold,
months later** — every command is copy-paste-able.

> Build tooling: **hatchling + uv** (see `pyproject.toml`). Release automation:
> `.github/workflows/publish_package.yml` (Trusted Publishing / OIDC).

---

## What is special about this package (read once)

| Thing | What | Why it matters |
|---|---|---|
| **✅ Dependencies now resolve** | `views-pipeline-core` reached PyPI on **2026-08-03** (3.0.0; now 3.2.0) | Historical note: 1.0.0 and 1.0.1 were published while this dependency had no PyPI release, so `pip install views-baseline` could not resolve *any* version (register C-38). That is cleared. **1.0.2 is the first release for which a clean-room install-back (§A) actually works — do it, since it has never been exercised.** |
| **Single package** | The `views-baseline` wheel ships one import package, `views_baseline` | `pip install views-baseline` → `import views_baseline`. |
| **Versions are write-once** | Once `X.Y.Z` is on PyPI it can never be re-uploaded or truly deleted (only "yanked") | Always **bump the version first**. For repeated TestPyPI rehearsals use a throwaway like `1.0.1.dev1`. |
| **uv + hatchling, NOT poetry** | Build backend is `hatchling.build`; tooling is `uv` | Use `uv build` / `uv publish`. |
| **Pure-Python wheel** | `py3-none-any`; `requires-python = ">=3.11,<3.15"` | No build cap; installs are fast. |

---

## TL;DR — release (the automated way)

Releases are published **by CI** when you publish a **GitHub Release** — you do **not** run
`uv publish` by hand. Auth is PyPI Trusted Publishing (no token); see
`.github/workflows/publish_package.yml`.

```bash
# 1. bump the version on a branch (you can NEVER reuse a published version)
$EDITOR pyproject.toml                          # under [project]: version = "X.Y.Z"
git commit -am "release: vX.Y.Z" && git push    # open a PR -> merge to main

# 2. cut the GitHub Release FROM main — this triggers the publish workflow:
gh release create vX.Y.Z --target main --title "views-baseline X.Y.Z" --notes "what changed"
#    (or GitHub UI: Releases -> Draft a new release -> tag vX.Y.Z on main -> Publish)

# 3. confirm: Actions tab shows "Publish Package" green, then
#    https://pypi.org/project/views-baseline/
```

The workflow guards the version (must beat PyPI), `uv build`s, and `uv publish`es via
Trusted Publishing — **no token needed**. First-ever setup requires the one-time PyPI
trusted-publisher config — see Prerequisites.

---

## Prerequisites (one-time setup) — Trusted Publishing

The release workflow authenticates with **Trusted Publishing (OIDC)** — there is **no
stored token**. A project owner enables it **once** on PyPI.

**Already done — this is history, not a step.** The project now exists on PyPI (1.0.0 published
2026-07-31, 1.0.1 on 2026-08-02), so the trusted publisher is a normal one and needs no further
setup. It was originally registered as a **pending** publisher, which is what PyPI offers for a
project that does not exist yet; the first OIDC publish created the project. Kept for the next
person setting up a *new* package from this template:

> PyPI → your account → **Publishing** → **Add a pending publisher (GitHub)**:
> - **PyPI Project Name:** `views-baseline`
> - **Owner:** `views-platform`  ·  **Repository:** `views-baseline`
> - **Workflow name:** `publish_package.yml`  ·  **Environment:** *(leave blank)*

After the first release creates the project, the same entry appears under the project's
**Settings → Publishing** as a normal trusted publisher. Until this is configured, the
workflow's publish step fails with an auth error — that's the only gap between merging the
workflow and it working.

> If you'd rather not use a pending publisher, do the **first** upload manually with a
> token (§B), then all future releases go through the automated path.

---

## A. TestPyPI dress rehearsal (optional)

```bash
rm -rf dist && uv build
uvx --from twine twine check dist/*            # both files must say PASSED
# sanity: the package is in the wheel
python3 -c "import zipfile,glob; ns=zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(); \
print('views_baseline/ present:', any(n.startswith('views_baseline/') for n in ns))"

# upload to TestPyPI (your terminal; replace the token — never paste it in chat)
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<YOUR-TESTPYPI-TOKEN> dist/*
```

> **Install-back is blocked for now:** `uv pip install views-baseline` needs
> `views-pipeline-core>=3.0.0`, which is on neither PyPI nor TestPyPI — so the clean-room
> install check cannot pass until pipeline-core is published. Until then, rehearsal is
> limited to `uv build` + `twine check` + upload. Re-enable the install-back step (with
> `--extra-index-url https://pypi.org/simple/` for `views-frames`) once pipeline-core is on
> PyPI.

---

## B. First real deployment — break-glass / manual (if not using a pending publisher)

```bash
git checkout main && git pull --ff-only
rm -rf dist && uv build && uvx --from twine twine check dist/*
# publish to REAL PyPI (the v1.0.0 tag already exists)
uv publish --token pypi-<YOUR-REAL-PYPI-TOKEN> dist/*
# confirm it's live:
curl -s https://pypi.org/pypi/views-baseline/json | \
  python3 -c "import sys,json;d=json.load(sys.stdin)['info'];print(d['name'],d['version'])"
```

After the first manual upload, switch to the automated path (§Prerequisites + TL;DR) for
every future release.

> 🔒 **Token safety:** type a token only in your own terminal; never paste it into a
> chat/transcript/PR. Prefix the command with a space (or `export UV_PUBLISH_TOKEN=…`) to
> keep it out of shell history. After the first publish, delete an over-privileged
> "entire account" token and rely on the tokenless Trusted-Publishing workflow.

---

## C. Future updates (the repeatable loop — automated)

1. **Bump `version`** in `pyproject.toml` under `[project]` (you cannot reuse a published
   version; SemVer — post-1.0 breaking = MAJOR).
2. Commit on a branch → PR → **merge to `main`**.
3. **Cut the GitHub Release from `main`** — triggers `publish_package.yml`:
   ```bash
   gh release create vX.Y.Z --target main --title "views-baseline X.Y.Z" --notes "what changed"
   ```
   It runs the **version guard**, `uv build`, `uv publish` via **Trusted Publishing**.
4. **Verify:** Actions → *Publish Package* green, then https://pypi.org/project/views-baseline/.

> Under the hood: `release: published` → `permissions: id-token: write` mints an OIDC token
> → PyPI checks the GitHub claim against the trusted publisher → upload. The version guard
> fails the run if `[project].version` isn't higher than what's on PyPI, so "forgot to bump"
> is a loud error, not a wasted version.

---

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `pip install views-baseline` can't resolve `views-pipeline-core` | Expected until pipeline-core 3.0.0 is on PyPI. Not a packaging bug in this repo. |
| `403 Forbidden` on the automated publish | Trusted publisher not configured (or name mismatch). Re-check Prerequisites (owner `views-platform`, repo `views-baseline`, workflow `publish_package.yml`). |
| `400 … File already exists` | That version is already uploaded — **versions are write-once**. Bump `version` and rebuild. |
| Version guard fails the run | `[project].version` ≤ current PyPI version. Bump it. |
| `twine check` fails on metadata | Stale build — `rm -rf dist && uv build` and re-check. |

---

## Provenance

- This guide and `.github/workflows/publish_package.yml` mirror the `views-frames` routine
  (`docs/guides/publishing-to-pypi.md`), adapted: a single wheel (`views_baseline`), a 3.11
  Python floor, and the pipeline-core dependency caveat above.
- **Not yet exercised by a real release** — the first `v1.0.0` publish (after the one-time
  PyPI pending-publisher config) confirms it; update this line when it does.

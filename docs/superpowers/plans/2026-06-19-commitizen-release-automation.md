# Commitizen Release Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive version bumps + `CHANGELOG.md` from Conventional Commits via Commitizen, gate PR titles, auto-open a release PR, and publish to PyPI on a deliberate merge — without weakening "no forced publish."

**Architecture:** `cz` (PEP 621 provider) owns the version + changelog. A PR-title workflow gates the Conventional-Commit signal (the repo squash-merges, so the PR title *is* the commit). `release.yml` (on push to `main`) opens/updates a `chore(release): vX.Y.Z` PR when `feat:`/`fix:` accumulate; merging it tags the commit and dispatches the existing standalone `publish.yml` (kept non-reusable for PyPI Trusted Publishing). The Actions bot uses only `GITHUB_TOKEN`; the maintainer's existing ruleset admin-bypass merges the bot's release PR.

**Tech Stack:** Python 3.11, `uv`, Commitizen (`cz_conventional_commits`), GitHub Actions, `amannn/action-semantic-pull-request`, PyPI Trusted Publishing (OIDC).

Spec: `docs/superpowers/specs/2026-06-19-commitizen-release-automation-design.md` · Issue [#59](https://github.com/uribrecher/audio-analysis-mcp/issues/59)

## Global Constraints

- **Python:** `requires-python = ">=3.11,<3.12"`; all tooling via `uv`.
- **Signed commits:** `main` ruleset requires signed commits. Commit locally with the Bash sandbox **disabled** (`dangerouslyDisableSandbox: true`) or signing silently no-ops. Verify each commit: `git cat-file commit HEAD | grep -c gpgsig` returns `1`.
- **Squash-only, PR-title is the commit:** the `main` ruleset enforces `allowed_merge_methods: ["squash"]`. The Conventional-Commit signal `cz` reads is the **PR title**.
- **`tag_format = "v$version"`** — must match `publish.yml`'s `tags: ["v*"]` *and* the PyPI Trusted Publisher (registered as `publish.yml`). **Do not rename `publish.yml`; do not touch the PyPI config.**
- **`publish.yml` must stay standalone** — never `workflow_call`/reusable (PyPI Trusted Publishing forbids reusable workflows).
- **`major_version_zero = true`** — stay in `0.x`.
- **CI uv-sources:** any `uv` call in CI must ignore `[tool.uv.sources]` (the `../SongFormer` editable redirect) via `UV_NO_SOURCES=1` or `--no-sources`.
- **This PR's own title must be `chore:`/`ci:`/`build:`** (not `feat:`/`fix:`) so it does not trigger a release after merge.
- **Branch:** all work on the current worktree branch; ship via the `personal-pr` skill (this repo tracks work as GitHub Issues).

---

### Task 1: Commitizen config + dev dependency

**Files:**
- Modify: `pyproject.toml` (add `[tool.commitizen]`; add `commitizen` to `[dependency-groups].dev`)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: a working `uv run cz` with `version_provider = "pep621"` reading `[project].version` (`0.1.0`), `tag_format = "v$version"`.

- [ ] **Step 1: Add the `[tool.commitizen]` block to `pyproject.toml`** (place it after the `[tool.pytest.ini_options]` block, before `[tool.mypy]`):

```toml
[tool.commitizen]
name = "cz_conventional_commits"
version_provider = "pep621"        # reads/writes [project].version
tag_format = "v$version"            # must match publish.yml tags: ["v*"] + the PyPI trusted publisher
update_changelog_on_bump = true
major_version_zero = true           # stay in 0.x; breaking changes bump minor until a deliberate 1.0
```

- [ ] **Step 2: Add `commitizen` to the `dev` dependency-group.** In `[dependency-groups].dev`, add the line (keep alphabetical-ish ordering near the other tools):

```toml
  "commitizen>=4.0",
```

- [ ] **Step 3: Lock + sync**

Run: `uv lock && uv sync --dev`
Expected: `uv.lock` updates; `commitizen` installs. (`--no-sources` not needed locally.)

- [ ] **Step 4: Verify cz reads the config**

Run: `uv run cz version --project`
Expected: prints `0.1.0`.

- [ ] **Step 5: Verify the dry-run infers NO release today** (the "no forced publish" anchor — only `docs:`/`chore:` exist since `v0.1.0`)

Run: `uv run cz bump --dry-run 2>&1; echo "exit=$?"`
Expected: NO `bump: version 0.1.0 → …` line; a "no eligible commits / no increment" message and a **non-zero** `exit=`. (If the installed `cz` prints a specific sentinel like `[NO_COMMITS_FOUND]`, note it — Task 5 relies on the non-zero exit, not the exact text.)

- [ ] **Step 6: Commit** (sandbox disabled for signing)

```bash
git add pyproject.toml uv.lock
git commit -m "build: configure Commitizen (pep621 version provider, v\$version tags)"
git cat-file commit HEAD | grep -c gpgsig   # must print 1
```

---

### Task 2: Backfill `CHANGELOG.md`

**Files:**
- Create: `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 1's `cz` config (`tag_format`, change-type map).
- Produces: a seeded `CHANGELOG.md` that `cz bump` appends to thereafter.

- [ ] **Step 1: Preview the generated changelog**

Run: `uv run cz changelog --dry-run`
Expected: a `## v0.1.0` (or `## 0.1.0`) section listing the historical `feat:`/`fix:` entries; non-conventional commits are skipped. No errors.

- [ ] **Step 2: Write the file**

Run: `uv run cz changelog`
Expected: `CHANGELOG.md` created at repo root.

- [ ] **Step 3: Sanity-check the content**

Run: `head -40 CHANGELOG.md`
Expected: a real changelog with the `0.1.0` `feat:`/`fix:` history. If it is empty or malformed, stop and inspect the change-type map rather than committing junk.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: seed CHANGELOG.md from Conventional-Commit history"
git cat-file commit HEAD | grep -c gpgsig   # must print 1
```

---

### Task 3: PR-title gate workflow

**Files:**
- Create: `.github/workflows/pr-title.yml`

**Interfaces:**
- Produces: a status check whose context is **`pr-title`** (job id), consumed by Task 6's ruleset update.

- [ ] **Step 1: Create `.github/workflows/pr-title.yml`**

```yaml
# .github/workflows/pr-title.yml
# Validates ONLY the PR title against Conventional Commits. The repo squash-merges, so the PR
# title becomes the commit on main that `cz bump` reads. Individual branch commits are unconstrained.
name: PR title
on:
  pull_request:
    types: [opened, edited, synchronize, reopened]

permissions:
  pull-requests: read

jobs:
  pr-title:                       # job id == required-status-check context "pr-title"
    runs-on: ubuntu-latest
    steps:
      - uses: amannn/action-semantic-pull-request@v5
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          types: |
            feat
            fix
            docs
            chore
            refactor
            perf
            test
            build
            ci
            style
            revert
```

- [ ] **Step 2: Validate the YAML** (if `actionlint` is available; otherwise a YAML parse)

Run: `command -v actionlint >/dev/null && actionlint .github/workflows/pr-title.yml || uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/pr-title.yml')); print('yaml ok')"`
Expected: no errors / `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/pr-title.yml
git commit -m "ci: gate PR titles on Conventional Commits"
git cat-file commit HEAD | grep -c gpgsig   # must print 1
```

---

### Task 4: Make `publish.yml` dispatchable

**Files:**
- Modify: `.github/workflows/publish.yml` (only the `on:` trigger block)

**Interfaces:**
- Produces: a `workflow_dispatch`-capable `publish.yml` that Task 5 invokes with `gh workflow run publish.yml --ref <tag>`. Stays standalone (NOT reusable).

- [ ] **Step 1: Replace the trigger block.** Change:

```yaml
on:
  push:
    tags: ["v*"]
```

to:

```yaml
on:
  push:
    tags: ["v*"]      # manual escape hatch: `cz bump` locally then `git push --follow-tags`
  workflow_dispatch:  # release.yml dispatches this (GITHUB_TOKEN-pushed tags don't fire push:tags)
```

Leave the `build` and `publish` jobs (including `environment: pypi` and OIDC) unchanged.

- [ ] **Step 2: Validate the YAML**

Run: `command -v actionlint >/dev/null && actionlint .github/workflows/publish.yml || uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml')); print('yaml ok')"`
Expected: no errors / `yaml ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish.yml
git commit -m "ci: allow publish.yml to be dispatched by the release workflow"
git cat-file commit HEAD | grep -c gpgsig   # must print 1
```

---

### Task 5: Release workflow (`release.yml`)

**Files:**
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: Task 1's `cz` config; Task 4's dispatchable `publish.yml`.
- Invariant: **"a `[project].version` with no matching `v<version>` git tag means a release commit just merged."** Self-correcting and loop-free: tagging the version satisfies the invariant on the next run; the `chore(release):` commit never triggers a bump.

- [ ] **Step 1: Confirm the exact `cz bump` no-commit/no-tag flag set** (behavior is version-dependent; verify before encoding it)

Run:
```bash
git switch -c _cz_probe
uv run cz bump --yes --files-only --changelog --dry-run 2>&1; echo "exit=$?"
git switch - && git branch -D _cz_probe
```
Expected: confirms `--files-only --changelog` is accepted and (with no eligible commits today) exits non-zero. If `--files-only` rejects `--changelog` on the installed version, use `--files-only` alone (with `update_changelog_on_bump = true` the changelog is still written) and adjust Step 2 accordingly.

- [ ] **Step 2: Create `.github/workflows/release.yml`**

```yaml
# .github/workflows/release.yml
# Commit-derived releases. On push to main:
#   Path A: [project].version has no matching tag  -> a release PR just merged -> tag + dispatch publish.yml
#   Path B: version already tagged                 -> if feat/fix accumulated, open/update the release PR
# The Actions bot uses only GITHUB_TOKEN; the maintainer merges the bot's release PR via admin bypass.
name: Release
on:
  push:
    branches: [main]

concurrency:
  group: release-main
  cancel-in-progress: false

permissions:
  contents: write        # push the release branch + version tag
  pull-requests: write   # open/update the release PR
  actions: write         # dispatch publish.yml (workflow_dispatch)

jobs:
  release:
    runs-on: ubuntu-latest
    env:
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      UV_NO_SOURCES: "1"           # ignore the ../SongFormer editable redirect on CI
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          fetch-tags: true

      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.11"

      # Run cz from an ephemeral env (no project install → no torch/songformer). Pin matches the
      # dev-group floor so CI and local `cz` behave identically.
      - name: Configure git identity
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

      - name: Read current version
        id: cur
        run: echo "version=$(uv run --no-project --with 'commitizen>=4.0' cz version --project)" >> "$GITHUB_OUTPUT"

      - name: Is the current version already tagged?
        id: tagged
        run: |
          if git rev-parse -q --verify "refs/tags/v${{ steps.cur.outputs.version }}" >/dev/null; then
            echo "value=true"  >> "$GITHUB_OUTPUT"
          else
            echo "value=false" >> "$GITHUB_OUTPUT"
          fi

      # ── Path A: release commit merged (version present, tag missing) → tag + publish ──
      - name: Tag release and dispatch publish
        if: steps.tagged.outputs.value == 'false'
        run: |
          V="v${{ steps.cur.outputs.version }}"
          git tag -a "$V" -m "$V"
          git push origin "$V"
          gh workflow run publish.yml --ref "$V"
          echo "Tagged $V and dispatched publish.yml"

      # ── Path B: version already tagged → open/update the release PR if a bump is due ──
      - name: Prepare or update the release PR
        if: steps.tagged.outputs.value == 'true'
        run: |
          git checkout -B release/next
          CZ_SPEC="commitizen>=4.0"   # quoted at every use so '>=' is never shell redirection
          set +e
          uv run --no-project --with "$CZ_SPEC" cz bump --yes --files-only --changelog  # adjust per Step 1 if needed
          code=$?
          set -e
          if [ "$code" -ne 0 ]; then
            echo "No bump-eligible commits since v${{ steps.cur.outputs.version }}; closing any stale release PR."
            gh pr close release/next --delete-branch 2>/dev/null || true
            exit 0
          fi
          W="$(uv run --no-project --with "$CZ_SPEC" cz version --project)"
          git add -A
          git commit -m "chore(release): v$W"
          git push -f origin release/next
          BODY="Automated release prepared by \`cz bump\`. Merging tags **v$W**; \`publish.yml\` then publishes to PyPI after the \`pypi\` environment approval. Generated — do not edit by hand."
          if gh pr view release/next --json number >/dev/null 2>&1; then
            gh pr edit release/next --title "chore(release): v$W" --body "$BODY"
          else
            gh pr create --base main --head release/next --title "chore(release): v$W" --body "$BODY"
          fi
```

- [ ] **Step 3: Validate the YAML**

Run: `command -v actionlint >/dev/null && actionlint .github/workflows/release.yml || uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('yaml ok')"`
Expected: no errors / `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add commit-derived release workflow (release PR + tag + dispatch publish)"
git cat-file commit HEAD | grep -c gpgsig   # must print 1
```

---

### Task 6: Apply GitHub repo/ruleset/environment settings

**Not committed to git** — these are GitHub-side config via `gh api`. Run after the workflows exist so the `pr-title` context is real. Use `$CLAUDE_JOB_DIR/tmp` for scratch files. Verify before/after; all changes are reversible.

- [ ] **Step 1: Squash commit subject = PR title, body = PR body**

Run:
```bash
gh api -X PATCH repos/uribrecher/audio-analysis-mcp \
  -f squash_merge_commit_title=PR_TITLE \
  -f squash_merge_commit_message=PR_BODY
gh api repos/uribrecher/audio-analysis-mcp --jq '{squash_title_default,squash_message_default}'
```
Expected: `{"squash_title_default":"PR_TITLE","squash_message_default":"PR_BODY"}`. (Squash-only is already ruleset-enforced; disabling merge/rebase is optional and skipped.)

- [ ] **Step 2: Add `pr-title` to the ruleset's required status checks** (preserve every existing rule)

Run:
```bash
RS="$CLAUDE_JOB_DIR/tmp/ruleset.json"
gh api repos/uribrecher/audio-analysis-mcp/rulesets/15346518 \
  --jq '{name,target,enforcement,conditions,bypass_actors,rules:
        (.rules | map(if .type=="required_status_checks"
           then .parameters.required_status_checks += [{"context":"pr-title","integration_id":15368}]
           else . end))}' > "$RS"
gh api -X PUT repos/uribrecher/audio-analysis-mcp/rulesets/15346518 --input "$RS"
gh api repos/uribrecher/audio-analysis-mcp/rulesets/15346518 \
  --jq '.rules[]|select(.type=="required_status_checks").parameters.required_status_checks[].context'
```
Expected last command lists: `test-and-lint`, `audit`, `packaging-smoke`, **`pr-title`**.

- [ ] **Step 3: Add the maintainer as a required reviewer on the `pypi` environment**

Run:
```bash
gh api -X PUT repos/uribrecher/audio-analysis-mcp/environments/pypi \
  --input - <<'JSON'
{"reviewers":[{"type":"User","id":695383}]}
JSON
gh api repos/uribrecher/audio-analysis-mcp/environments/pypi --jq '.protection_rules'
```
Expected: a `required_reviewers` protection rule listing user `uribrecher`.

---

### Task 7: Document the release flow

**Files:**
- Modify: `README.md` (add a `## Releases` section, after `## Development`)
- Modify: `CLAUDE.md` (add a short release/PR-title note)

- [ ] **Step 1: Add `## Releases` to `README.md`** (insert after the `## Development` block, before `## Scratch tools`):

```markdown
## Releases

Versioning and the changelog are **derived from Conventional Commits** by
[Commitizen](https://commitizen-tools.github.io/commitizen/) — never hand-edited. Because the repo
squash-merges, **the PR title is the release signal** and is gated by the `pr-title` check
(`feat:` → minor, `fix:` → patch; `docs:`/`chore:`/etc. → no release). We stay in `0.x`
(`major_version_zero`).

**Normal flow (automatic):**
1. Merge feature PRs with Conventional-Commit titles.
2. When `feat:`/`fix:` have accumulated, `release.yml` opens/updates a **`chore(release): vX.Y.Z`**
   PR that bumps `[project].version` and regenerates `CHANGELOG.md`.
3. Merge that release PR (admins bypass the ruleset to merge the bot PR). `release.yml` then tags
   `vX.Y.Z` and dispatches `publish.yml`.
4. Approve the **`pypi`** environment deployment. `publish.yml` builds and publishes to PyPI via
   Trusted Publishing.

**Manual escape hatch:** `uv run cz bump` locally (bumps version + changelog + tag), then
`git push --follow-tags` — the pushed tag fires `publish.yml` directly.
```

- [ ] **Step 2: Add a note to `CLAUDE.md`** (append a short section after `## Testing`):

```markdown
## Releases

Versions + `CHANGELOG.md` are commit-derived via Commitizen — see README "Releases". **Every PR
title must be a valid Conventional Commit** (gated by the `pr-title` check); the squash-merged title
is what `cz bump` reads. Releases are cut by merging the auto-generated `chore(release):` PR, then
approving the `pypi` environment. Never hand-edit `[project].version` or `CHANGELOG.md`.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the Commitizen release flow"
git cat-file commit HEAD | grep -c gpgsig   # must print 1
```

---

## Verification (whole-plan, before opening the PR)

- [ ] `uv run cz version --project` → `0.1.0`; `uv run cz bump --dry-run` → no increment, non-zero exit (no forced publish).
- [ ] `uv run pytest -m "not slow" -q` and `uv run mypy src/` still pass (config-only change shouldn't affect them).
- [ ] `CHANGELOG.md` exists with real history.
- [ ] All four workflow files parse (actionlint/YAML).
- [ ] Ruleset lists `pr-title` among required checks; `pypi` env has a required reviewer; squash subject = PR title.
- [ ] **This PR's title is `chore:`/`ci:`/`build:`** (not `feat:`/`fix:`) so the first post-merge `release.yml` run is a no-op.

## First-run watch (after this PR merges — not part of the PR)

- On merge, `release.yml` runs Path B: version `0.1.0` is tagged → `cz bump` finds no `feat:`/`fix:` → **no release PR**. Confirm it's a clean no-op.
- The next `feat:`/`fix:` PR should produce a `chore(release):` PR. Watch that first real release end-to-end (merge → tag → `pypi` approval → PyPI).

## Risks

- **`release.yml` is bespoke** — the single "untagged version = release" invariant keeps it small; the manual `cz bump` path remains as fallback; watch the first live release.
- **Ruleset edit (Task 6 Step 2)** must preserve all existing rules — the `jq` transform only appends one context; verify the after-state lists all four checks.
- **PyPI env binding** — if the trusted publisher pins an environment ≠ `pypi`, the first publish fails with an OIDC error; the PyPI config is left untouched (already correct for filename `publish.yml`).

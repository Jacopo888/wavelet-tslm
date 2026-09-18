# Pinning prime-rl to work around the `deps/research-environments` submodule drift

## Problem

`scripts/setup.sh` clones `https://github.com/PrimeIntellect-ai/prime-rl.git` and then runs:

```bash
git -c url."https://github.com/".insteadOf="git@github.com:" \
  submodule update --init \
    deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
```

On the current `main` branch of prime-rl (tip as of this writing:
`3758321f08a953498893a2a02f022f1179d9a7ed`, "fix: preserve final eval during
teardown (#3324)", 2026-08-24) this fails because `.gitmodules` no longer
declares a `deps/research-environments` submodule. Commit
`4308fe3aba066d346dd51f447cd9381860a33103` — "chore!: point at the renamed
prime-envs repo (#3241)" (2026-08-11) — renamed the submodule from
`research-environments` (`deps/research-environments`) to `prime-envs`
(`deps/prime-envs`). `git submodule update --init deps/research-environments`
on any commit at or after that rename errors out because the path is not
registered in `.gitmodules` any more.

## Chosen pin

```
SHA:  da53d4c064eb964ebb08497d9a236eeb1fa50c83
Date: 2026-08-11 12:23:08 -0700
Subject: chore: bump renderers for Nemotron 3.5 (Lightning) renderer (#3239)
```

This is the **direct (sole) parent** of the rename commit `4308fe3ab` — i.e.
the last commit reachable from `main` whose `.gitmodules` still defines all
four submodules our setup needs:

```
[submodule "verifiers"]           path = deps/verifiers
[submodule "renderers"]           path = deps/renderers
[submodule "research-environments"] path = deps/research-environments
[submodule "pydantic-config"]     path = deps/pydantic-config
```

### Why this SHA and not an earlier one

Walking `.gitmodules` history (`git log --oneline --follow -- .gitmodules`)
shows the file was edited 7 times. The most recent edit that *set* all four
paths at once was `632ef2b31820408a874332dbfa7d6a416ed77583` ("chore: remove
private configs submodule (#2741)", 2026-06-09). That commit is a valid
answer too, but it is stale: `.gitmodules` was not touched again until the
rename 237 commits later, so every one of those 237 commits — up through
`da53d4c06` — still has the exact same, working `.gitmodules` content. Pinning
to `da53d4c06` instead of `632ef2b31` gets ~2 months of additional fixes/
features on the exact same submodule layout, at zero extra risk. `da53d4c06`
is therefore the better pin: it is the true "last commit on `main` before the
submodule set drifted," not just the last commit that happened to *write* to
`.gitmodules`.

(`configs/private` / `research-configs` also disappears earlier, at
`632ef2b31`, but it was never one of the 4 paths `setup.sh` requests, so it is
irrelevant here.)

## Reproducing the pin on the target Linux machine

```bash
# Fresh clone: pin directly.
PRIME_RL_REPO=https://github.com/PrimeIntellect-ai/prime-rl.git \
PRIME_RL_REF=da53d4c064eb964ebb08497d9a236eeb1fa50c83 \
  bash scripts/setup.sh

# Equivalent, done by hand (also what to do if ./prime-rl already exists
# from a prior failed run — setup.sh only checks out PRIME_RL_REF on a fresh
# clone, so an existing checkout out needs this run manually):
cd prime-rl
git fetch origin da53d4c064eb964ebb08497d9a236eeb1fa50c83   # if not already present locally
git checkout da53d4c064eb964ebb08497d9a236eeb1fa50c83
git -c url."https://github.com/".insteadOf="git@github.com:" \
  submodule update --init \
    deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
```

Verified locally (Windows, git-only, no GPU) on 2026-08-25:
`git checkout da53d4c06` succeeded, and the four submodules cloned and
checked out cleanly:

```
$ git submodule status deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
 4f5ae373582ceffdbf7e6bd1998c9ad568fcc1ad deps/pydantic-config (v0.4.2)
 2846a3dcd29318c1fc98de3498bab4190997af9e deps/renderers (renderers-v0.1.9-4-g2846a3d)
 c0a0d1d7ac011a60c7cd90ce0feb5e12854ce2a1 deps/research-environments (c0a0d1d7)
 d53ab56f0cd46ff6bce5c5965df1c61678d800c7 deps/verifiers (harnesses-v0.1.2-477-gd53ab56f)
```

All four are populated (no `+`/`-` prefix, i.e. checked-out commit matches
the index) and all four repos are in fact public — they clone anonymously
over HTTPS with no credentials, which is why `setup.sh`'s
`insteadOf="git@github.com:" -> "https://github.com/"` rewrite is enough; no
SSH key / deploy key is required on the target machine.

Operational note observed during verification: if the submodule-update
command is interrupted partway (e.g. a transient network failure on one of
the four clones), an *already-cloned-but-not-yet-checked-out* submodule can
be left with an empty working tree (`git submodule status` shows a leading
`+`, and `git -C deps/<name> status` shows everything as "deleted" even
though the objects are present). Fix: just re-run the same
`git submodule update --init <path>` for that one path — it's idempotent and
finishes the checkout without re-cloning.

## What still needs to happen on the Linux/GPU machine

Only the git/submodule part was validated here (this machine is Windows,
git-only, no CUDA/GPU — installing the actual `prime-rl` Python package was
explicitly out of scope per the task). Once `prime-rl` is checked out at
`da53d4c06` with the four submodules populated as above, resume `setup.sh`
from step 3 onward:

1. **`uv sync --all-extras`** inside `prime-rl/` — installs the trainer +
   inference + verifiers dependency set (heavy step; needs a CUDA toolchain
   for flash-attn / kernel builds).
2. **Install TimeRLM's own packages editable into prime-rl's venv**:
   ```bash
   uv pip install -e "${GEN_DIR}"                 # anomalyXL
   uv pip install --no-deps -e "${ENV_DIR}"        # environments/timeseries_qa
   uv pip install --no-deps -e "${ENV_V1_DIR}"     # environments/timeseries_qa_v1
   ```
   (`--no-deps` on the env packages matters: a plain `-e` install does not
   honor their `[tool.uv.sources]` pins and would re-resolve `verifiers` from
   PyPI, clobbering the editable `deps/verifiers` submodule the RL path
   depends on.)
3. **Verify imports**: `uv run python -c "import anomalyxl, timeseries_qa, timeseries_qa_v1; print('imports OK')"`.
4. **Create the `rlm-harness` venv**:
   ```bash
   uv venv rlm-harness/.venv
   uv pip install --python rlm-harness/.venv -e rlm-harness
   ```
5. Proceed to the eval/RL commands in `docs/REPRODUCE.md`.

If `prime-rl` upstream later re-adds a stable `deps/research-environments`
path (or `setup.sh` is updated to request `deps/prime-envs` instead), this
pin can be dropped and `setup.sh` run again against `main` unmodified.


## ADDENDUM (2026-09-03, wavelet-tslm L2 kit): this pin is NOT API-compatible with this repo

`da53d4c06` was chosen for its `.gitmodules` layout only. Its `deps/verifiers` (d53ab56f,
2026-08-10) post-dates two breaking v1 API changes that `environments/timeseries_qa_v1`
relies on — `RolloutContext` renamed to `ModelContext` (2026-07-06, #1940) and per-task
behaviour moved from `Taskset` onto `Task` (2026-07-10, #1948) — so
`import timeseries_qa_v1` fails at this pin. prime-rl itself also dropped the schema of
`configs/rl/anomalyxl_rlm_v5_ablation_fixed.toml` (`orchestrator.advantage`,
`orchestrator.student.client`, `[[orchestrator.train.env]]`, `max_inflight_rollouts`) on
2026-06-27 (#2746) and 2026-07-29 (#3150/#3156).

Verified commit by commit (submodule pointer + grep of every imported symbol): the last
upstream commit where the env imports AND the paper config load unmodified is

```
SHA:  c5afcaa2e2db6751e0f2eae437afa6bd532fa87f
Date: 2026-06-26
Subject: feat: default ep to auto for MoE models (#2868)
deps/verifiers @ aa95826d (2026-06-25)
```

Same four submodules, same `torch==2.11.0+cu128` / `flash-attn 2.8.3`, vLLM 0.23.0.
`wavelet-tslm/l2/node_l2_env.sh` installs this commit with `uv sync --frozen`.

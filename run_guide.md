# AGS `AGS-Injection-v1` + SOCT — Quickstart

Get from clone to a trained-policy run in a few commands. Setup, run, configuration, and results
only — the reward/state/action design and the physics results live elsewhere.

**What you're running:** a TD3 agent (SOCT `KerasTD3-v0`) tuning 96 dipole-corrector currents in a
Julia beam-tracking simulation of the AGS injection ring (env `AGS-Injection-v1`, a
single-step optimizer) to maximize beam survival across all 73 BPMs.

## 1. Layout

Two repos, side by side. **Everything AGS-specific lives in `AGS-sim-env/`** — the Julia
environment, the gym wrapper (`ags_env.py`), the Gymnasium registration module
(`ags_gymnasium.py`), the launcher (`soct_launcher.py`), and the policy checker
(`verify_policy.py`); `pyproject.toml` makes it a pip-installable package.
`SciOptControlToolkit` is used **unmodified** from upstream `main`:

```
./                    # any workspace dir
├── AGS-sim-env/              # the environment (Julia project + Python glue, all tracked here)
│   ├── pyproject.toml        #   pip-installable (editable!), gymnasium entry points, ags-soct-run
│   ├── CondaPkg.toml         #   pins CondaPkg's embedded Python for PythonCall (§2.3)
│   ├── ags_env.py            #   gym wrapper — ALL RL glue: state/reward/action/knobs (§4)
│   ├── ags_gymnasium.py      #   import-once Gymnasium registration (gym.make works after)
│   ├── soct_launcher.py      #   pin Julia → import juliacall → register env → run SOCT driver
│   ├── verify_policy.py      #   load any checkpoint vs zero-action; prints reward + S survival
│   └── src/
│       ├── AGS_GymEnv.jl     #   THIN facade: include(tracking.jl) + primitives only
│       ├── tracking.jl       #   PHYSICS single source: machine, beam, tracking kernel
│       └── ags_apertures.jl  #   lattice/apertures/power-supplies (included by tracking.jl)
└── SciOptControlToolkit/     # SOCT   (agents, driver, buffers — untouched)
```

Prerequisites: [conda](https://docs.conda.io/) and [juliaup](https://github.com/JuliaLang/juliaup)
— don't assume either is installed or on `PATH`; probe first:

```bash
conda --version && juliaup --version && julia --version
```

- No conda: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and run
  `~/miniconda3/bin/conda init bash`, then restart the shell — an installed-but-uninitialized
  conda is invisible to §2.4.
- No juliaup/julia: `curl -fsSL https://install.julialang.org | sh -s -- --yes`, then
  `. ~/.bashrc` (`--yes` = non-interactive). A fresh install defaults to `release` (1.13 now);
  §2.2 re-pins it to 1.12.

## 2. Setup

### 2.1 Clone

```bash
cd <workspace-dir>
git clone https://github.com/eiad-hamwi/AGS-sim-env.git
git clone https://github.com/JeffersonLab/SciOptControlToolkit.git
cd AGS-sim-env && git checkout 1-gym-env && cd ..    # branch with the gym env facade over tracking.jl
```

Keep both as real `git clone` directories — the driver reads a git hash from the CWD at startup.

### 2.2 Julia toolchain (do this before anything Python)

```bash
juliaup add 1.12            # fresh juliaup ships only 'release' — 'default 1' errors until added
juliaup default 1.12        # 1.12.x — NOT 'release'/'1' (that is 1.13.x now), NOT lts 1.10
julia --version             # must print 1.12.x
```

Why it matters: the Julia deps (BeamTracking/Beamlines stack) only load under Julia ≥ 1.12, and
`soct_integration.py` embeds juliacall from `~/.julia/juliaup/julia-1.12*` /
`libjulia.so.1.12*` **only** — a newer default (today's `release` = 1.13.1, which plain
`juliaup default 1` also resolves to) fails at registration with `RuntimeError:
soct_integration: no Julia 1.12 under ~/.julia/juliaup`. juliacall embeds whatever juliaup's
*default* channel points at, so `default 1.12` must stick. The older symptom of the same
mismatch (e.g. the `lts` 1.10 default): `ERROR: could not load library
".../.juliaup/bin/../lib/julia/sys.so"` at `import juliacall` — that error is this version
problem, not a Python problem.

### 2.3 Julia project dependencies (once per machine)

`Manifest.toml` is committed, so versions (including the git-URL `BeamDistributions` fork and
`BeamTracking 0.7.1`) are pinned exactly — just instantiate:

```bash
cd AGS-sim-env
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Expect several minutes of precompilation. The Conda/pixi environment (`.CondaPkg/`,
gitignored) is created during this instantiate (precompile of AGS_GymEnv) or on the
first Julia *run* inside PythonCall, whichever comes first — one-time, a few minutes.

`CondaPkg.toml` (committed) pins the embedded Python: `[deps] python = "3.13.*"`. Keep it.
CondaPkg's implicit spec is a bare `python`, which on 2026-10 conda-forge resolves to the
**3.15 alpha — a build with no shared libpython** — and precompile of `AGS_GymEnv` /
`BeamlinesPythonCallExt` then dies with `InitError: Could not find Python library for Python
executable ".../.CondaPkg/.pixi/envs/default/bin/python"` ("during initialization of module
C"). Spec form matters: CondaPkg merges this spec with PythonCall's own range into one pixi
`version` string, so `"3.13.*"` parses, `"==3.13.*"` makes pixi exit 1 ("expected a version
specifier but looks like a matchspec"), and bare `"3.13"` only works behind pixi's "ambiguous
version specifier" warning (which a future pixi turns into an error). Keep the spec in range
with `env.yaml`'s python.

### 2.4 Conda environment + SOCT install

```bash
cd SciOptControlToolkit
conda env create -f env.yaml -n ags_rl          # or keep default name jlab_opt_control
conda activate ags_rl                           # non-interactive: `conda run -n ags_rl <cmd>`
pip install -e .
pip install juliacall==0.9.35                   # MUST match the Manifest's PythonCall —
                                                # 0.9.36 fails "PythonCall.jl did not start
                                                # properly"; keep in sync with Manifest.toml
pip install -e ../AGS-sim-env                   # the AGS env package — MUST be editable:
                                                # the Julia project lives in that checkout
```

`conda env create` installs several GB of wheels (torch + TF + CUDA stack) — budget ~15 min
and **wait for it to exit 0 before the pip steps below**. An interrupted pip section leaves a
half-built env whose first driver run dies with `ModuleNotFoundError: No module named
'tensorflow'` — that is an unfinished install, not a broken SOCT. Gate on it before continuing:

```bash
python -c "import tensorflow, torch, gymnasium; print('env ok')"
```

Equivalent if you prefer to skip env.yaml: `conda create -n ags_rl python=3.13` followed by
`pip install -r requirements.txt` (env.yaml's pip section and requirements.txt are the same
list). Either way use python 3.13 to match the tested configuration.

Then register AGS inside SOCT — one command (idempotent; also accepts an explicit
checkout path):

```bash
ags-install-soct                # adds the guarded `import ags_gymnasium` blocks
```

It inserts `import ags_gymnasium` (inside `try/except ModuleNotFoundError`) at the
top of `jlab_opt_control/__init__.py` and above `import tensorflow` in the driver
script. The driver resolves `AGS-Injection-v1` like a built-in then — the `gym`
branch of `create_and_configure_env`. Undo: `git checkout` those two files.

### 2.5 Verify

A cheap wiring check first — it starts Julia but never touches TensorFlow, so import ordering
is safe (and it isolates "Julia side broken" from "TF side broken"):

```bash
python -c "import ags_gymnasium; print('ags_gymnasium import OK')"
```

Then the real smoke run:

```bash
cd SciOptControlToolkit                # any cwd works; logdirs/paths are self-located
python -m jlab_opt_control.drivers.run_continuous \
    --env AGS-Injection-v1 --nepisodes 2 --logdir /tmp/ags_smoke
```

Look for `Created gym environment: AGS-Injection-v1`, `Size of State Space -> 315`,
`Size of Action Space -> 96`. The default machine (mm σ = 3.4e-3, lattice seed
20261022) is a **hard-but-learnable lattice**: zero-action reward **≈ 40**
on this seed (~55 median over fresh draws), good play reaches 95–100. Expect the
2-episode smoke to land **~30–60** (random/early-TD3 actions are net harmful at
this σ) — 315/96 plus a value in that band = wiring success (§5). The legacy
`AGS-Injection-v0` (pre-2026-10-21, −1000·RMS centering reward) is NOT
registered and its logs are not comparable. First run compiles
the Julia stack (a few minutes); dead-BPM episodes measure fast (~40–100 ms).

Expected noise in any run's log (harmless, CPU fallback family):
- `ERROR:CfgLogger:Problem with key request using None/2` — known SOCT cfg-logger quirk.
- `Could not find cuda drivers` / `INTERNAL: CUDA Runtime error: ... cudaGetRuntimeVersion`
  — TF falls back to CPU, fine (see §6).
- possibly `Warning: ... ML4SNSX_BPMMinimizationEnv ...` — unrelated PACES probe.

**Why it must sit above TensorFlow:** embedding libjulia after TF's runtime
segfaults the process (SIGSEGV — reproduced even with the Julia env vars pre-staged
and with `JULIA_NUM_THREADS=1`), and `import jlab_opt_control` loads TF during
package import (its `__init__` chain reaches Keras). `import ags_gymnasium` in the
package `__init__` is therefore the only ordering-correct hook inside SOCT;
`ags_gymnasium` pins juliacall to the newest `~/.julia/juliaup/julia-1.12*` (export
`PYTHON_JULIACALL_EXE`/`LIB` to override) and starts Julia in this project. Prefer
zero SOCT edits? `ags-soct-run` (or `python ../AGS-sim-env/soct_launcher.py`) is a
self-locating launcher that performs the same ordering and registration outside.
With SOCT already patched the launcher resolves through the gym registry too, so
it prints `Created gym environment` instead of `Created custom environment` — both
mean success; the `custom` marker only appears on an unpatched checkout.

## 3. Running

### 3.1 Canonical run command

```bash
cd SciOptControlToolkit                  # git hash + relative logdirs are CWD-based
conda activate jlab_opt_control
python -m jlab_opt_control.drivers.run_continuous \
    --env AGS-Injection-v1 --agent KerasTD3-v0 --nepisodes 10000 \
    --inference_interval 50 --nepisode_avg 20 \
    --logdir results/ags_td3
```

(Requires the `__init__.py` block above — or run through the `ags-soct-run` fallback instead.)

Expect: first episode ~40 s (Julia first-compile inside the process), then ~0.3–1.5
s/episode → **10k episodes ≈ 1 hour** on CPU. Under the mm model the ring measurement
itself is only ~40–100 ms (the kernel breaks early once the beam dies), so TF updates
now take the larger share of episode time. First 2500 episodes (`warmup_size`) are pure
random exploration — the reward only starts moving after that.

Stale reference (pre-refactor quad-error env): the 2026-10-08 run went −117.6 → −107.
Those numbers do NOT apply to the main-magnet model — re-baseline from §5 instead.

### 3.2 Common flags

| flag | default | note |
|---|---|---|
| `--env` | `Pendulum-v1` (SOCT default) | pass `--env AGS-Injection-v1`; other SOCT/gym envs still reachable |
| `--agent` | `KerasTD3-v0` | |
| `--nepisodes` | `100` | episodes; each episode is **one env step** (single-step task) |
| `--nsteps` | `-1` | leave at −1; multi-step TimeLimit is meaningless here |
| `--inference_interval` | `10` | deterministic (noise-free) evaluation every N episodes — the real signal |
| `--nepisode_avg` | `20` | averaging window for logging/checkpoint gating |
| `--model_save_threshold` | `0.05` | checkpoint gate on avg inference-reward improvement |
| `--index` | `0` | prefixes the run folder name |
| `--logdir` | `./results/` | parent dir; run folder = `index<NNN>_env_<...>_agent_<...>_hash_<githash>_time_<stamp>/` |
| `--inference` | off | evaluation-only mode |
| `--load_model` | `None` | checkpoint dir to load (combine with `--inference` to evaluate a saved policy) |
| `--agent_cfg` / `--actor_cfg` / `--critic_cfg` / `--buffer_cfg` | packaged cfgs | absolute-path cfg variants |

Agent hyperparameters: `jlab_opt_control/cfgs/keras_td3.cfg` — `warmup_size=2500,
batch_size=256, lr=3e-4, tau=0.005, discount=0.99, ER-v0 buffer, FCNN actor+critic`. Any run
shorter than 2500 episodes acts randomly — a 100-episode run only verifies wiring.

## 4. Environment configuration

**Architecture:** the physics single source of truth is `src/tracking.jl` (machine via
`ags_apertures.jl`, beam, and the tracking kernel `measure_observables!`), kept identical to
the physics branch. `src/AGS_GymEnv.jl` is a **thin facade** that `include`s it and exposes
only marshalling primitives (`redraw_misalignments!`, `set_currents!`, `zero_currents!`,
`set_tunes!`, one-pass `measure`). Everything RL-specific — state assembly, reward, action
layout/bounds, episode logic, knobs — lives in `ags_env.py`. **Updating physics = updating
`tracking.jl` only; the env picks it up on the next recompile.**

There is **no env cfg file**. Knobs (all Python-side, `ags_env.py`):

| knob | where | default | effect |
|---|---|---|---|
| `misalign_sigma` | `AGSEnv.__init__` (or `register(...)` kwargs) / env var `AGS_SIGMA_TASK` | `3.4e-3` | main-magnet x misalignment σ (m) over the six `mm_*` groups, drawn **once at env construction** — frozen for the run, from the RNG seeded by `lattice_seed`. **3.4e-3 = task point 2026-10-31** (zero-action ~55% survival median over draws — ≈40 on seed 20261022; RL 99+ at ±50 A — study doc §9). `1e-2` = tracking.jl physics default: death lattice (§7). `1.5e-3` archived (§6); ≤1e-3 lossless (max-S trivial) |
| `h_tune`, `v_tune` | same | `0.0` | fixed tune-control settings (not actions) |
| `bpm_noise` | same | `(0, 0, 0)` | electrical RMS noise on S, Dx, Dy (zero = deterministic) |
| `lattice_seed` | same | `20261022` | `Random.seed!` on the Julia RNG before the lattice draws — same seed = same lattice (benchmark continuity; §9 eval protocol draws fresh seeds per eval) |
| `norm_action` | same | `True` | action space `[-1, 1]`, scaled by `ACTION_BOUND` inside `step()`; `False` = raw ampere box |
| corrector bounds | `ags_env.py` `ACTION_BOUND` / env var `AGS_ACTION_BOUND` | ±50 A | action box scale — learning optimum (§9: 25–100 A workable, ≥500 A collapses) |
| reward | `ags_env.py` `_reward` | — | `100·mean(S/s_full) + SHAPE_W·(1 − min(RMS/rms_ref, SHAPE_CLIP))`, range 0–100, max = full S; `SHAPE_W=0.1`, `SHAPE_CLIP=3`; `s_full`/`rms_ref` measured per lattice |
| beam params | `src/tracking.jl` beam block (physics-side, by design) | — | emittance, centroid, dispersion, population |

- **State (315)** — layout unchanged across the refactor:
  73 BPMs × (S, Dx, Dy) sorted by BPM name + 48 `I_dhc` + 48 `I_dvc` currents.
  `S` = charge·alive_fraction·cal (10.014 = full survival); `Dx/Dy` = surviving-bunch centroid × S.
- **Action (96)** — normalized `[-1, 1]` (× ±50 A) by default: **block layout** —
  first 48 = sorted `dhc`, next 48 = sorted `dvc` (the `AGSEnv.step` mapping in
  `ags_env.py` is authoritative). Raw-amp checkpoints from the v0 era are NOT
  transferable to the normalized space.
- **Objective / reward**: MAX OUT BPM signal S along the whole line —
  `100 · mean(S/s_full) + 0.1·(1 − min(RMS/rms_ref, 3))` ∈ [0, 100]. Losing
  beam can never raise the reward; the centering term only orders
  full-survival plateau configs (where the staircase S has no gradient).
  The legacy −1000·RMS centroid reward is superseded (it scored total loss
  ~0 above every live state). At the default 3.4e-3 lattice the S-greedy
  ceiling is ~99 and trained TD3 reaches 99.3–99.8 (median of last 4
  checkpoints, lattice 20261022 — study doc §9).

## 5. Reviewing the results

A run folder contains:

- `metrics/` — TensorBoard event files (`Training Reward`, `Inference Reward` scalars)
- `models/` — checkpoints, 6 Keras `.weights.h5` each: `init`, periodic `epoch_NNNNN` (every
  1000 eps), and improvement-triggered `epoch_NNNNN_PPP` (PP = avg-inference improvement %). The
  improvement saves fire **whenever the running best improves — expect hundreds** (the 10k
  reference run wrote ~212). The best policy is usually the last `epoch_NNNNN_PPP`; there is no
  separate `best/` dir.
- `buffers/buffer.npy` — replay snapshot (**~5.8 GB at 10k episodes** — prune when archiving)
- `results.npy` — per-episode training rewards
- `cfgs/` — resolved component configs

### TensorBoard

```bash
tensorboard --logdir results/          # or one run dir; port 6006; SSH-tunnel if remote
```

**Baselines at the current default (3.4e-3, lattice 20261022, study doc §9):**
zero action ≈ 40 on this seed (≈ 54.6 median over fresh draws); S-greedy ≈ 98;
TD3 from scratch at ±50 A: **99.3–99.8**
(both arms; the ±1 normalized-action policy is the shipped default). Headline
statistic MUST be the median over the last 4 periodic checkpoints — single
checkpoints are phase lotteries (one 500 A chain scored 4 vs 80 across
snapshots) — evaluated on fresh lattice draws, with the loader guard
(KerasTD3.load silently scores a fresh actor when a load fails; assert
`Models loaded successfully` per checkpoint). Historical note: at the old
1e-2 default the machine was unrecoverable (zero-current ladder within +0.2
pt at every authority; TD3 landed exactly on do-nothing) — that regime is
kept in §7 of the study doc and explains the pre-v1 logs.

### Evaluating a saved policy

```bash
python ../AGS-sim-env/soct_launcher.py --inference --nepisodes 3 \
    --load_model <run>/models/epoch_10000 --logdir /tmp/eval
```

`AGS-sim-env/verify_policy.py` additionally loads every checkpoint dir you point it at and prints
per-checkpoint reward vs zero-action plus the min/max S survival check
(`python ../AGS-sim-env/verify_policy.py <run_dir>` from SOCT).

Note: lattice draws are **seeded**: `AGSEnv(lattice_seed=20261022)` reseeds
Julia's global RNG before the redraw, so same seed = same frozen lattice —
retraining, verification, and `--inference` reproduce each other. Per-§9
evaluation on *fresh* draws, pass a different `lattice_seed` per eval process.
Checkpoint-to-checkpoint comparisons must still share one seed.

## 6. Notes

- TF runs CPU-only out of the box in this conda env (its CUDA wheels vs system driver mismatch);
  batch 256 nets are small, CPU ≈ fine. `Could not find cuda drivers` warnings are expected.
- Everything relative (run dirs, git hash) keys off the CWD you launch from — stay inside
  `SciOptControlToolkit/`.
- Episode clock note: `AGSEnv` (`ags_env.py`) runs the `tracking.jl` kernel once per
  `reset()`/`step()` (single-pass sampling, ≲0.1 s — faster when the beam dies early);
  with the mm model the TF agent update, not the measurement, is often the bottleneck.
- Three Julia/Python mismatch failure families — all §2.2–2.3 setup problems, not code bugs:
  (1) `sys.so ... cannot open` → the default channel is not 1.12 (§2.2); (2) `soct_integration:
  no Julia 1.12 under ~/.julia/juliaup` → only 1.13+ is installed (§2.2); (3) `Could not find
  Python library ... .CondaPkg/...` → missing or stale `CondaPkg.toml` pin (§2.3). The
  launcher pins (§2.5) override auto-detection for (1) and (2).

# AGS `AGS-Injection-v0` + SOCT — Quickstart

Get from clone to a trained-policy run in a few commands. Setup, run, configuration, and results
only — the reward/state/action design and the physics results live elsewhere.

**What you're running:** a TD3 agent (SOCT `KerasTD3-v0`) tuning 96 dipole-corrector currents in a
Julia beam-tracking simulation of the AGS injection ring (env `AGS-Injection-v0`, a
single-step optimizer) to center the beam at all 73 BPMs.

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
│   ├── ags_env.py            #   gym wrapper (loads src/AGS_GymEnv.jl through juliacall)
│   ├── ags_gymnasium.py      #   import-once Gymnasium registration (gym.make works after)
│   ├── soct_launcher.py      #   pin Julia → import juliacall → register env → run SOCT driver
│   ├── verify_policy.py      #   load any checkpoint vs zero-action; prints reward + S survival
│   └── src/AGS_GymEnv.jl     #   the environment proper (state/action/reward)
└── SciOptControlToolkit/     # SOCT   (agents, driver, buffers — untouched)
```

Prerequisites: [conda](https://docs.conda.io/) and [juliaup](https://github.com/JuliaLang/juliaup)
(`curl -fsSL https://install.julialang.org | sh`).

## 2. Setup

### 2.1 Clone

```bash
cd <workspace-dir>
git clone https://github.com/eiad-hamwi/AGS-sim-env.git
git clone https://github.com/JeffersonLab/SciOptControlToolkit.git
cd AGS-sim-env && git checkout 1-gym-env && cd ..    # branch carrying the gym env + updated tracking.jl
```

Keep both as real `git clone` directories — the driver reads a git hash from the CWD at startup.

### 2.2 Julia toolchain (do this before anything Python)

```bash
juliaup default 1          # 1.12.x — NOT the lts default 1.10
julia --version            # must print 1.12.x
```

Why it matters: the Julia deps (BeamTracking/Beamlines stack) only load under Julia ≥ 1.12, and
`juliacall` embeds whatever juliaup's *default* channel points at. On the default `lts` install
you get `ERROR: could not load library ".../.juliaup/bin/../lib/julia/sys.so"` at
`import juliacall` — that error is this mismatch, not a Python problem.

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

Then register AGS inside SOCT — one command (idempotent; also accepts an explicit
checkout path):

```bash
ags-install-soct                # adds the guarded `import ags_gymnasium` blocks
```

It inserts `import ags_gymnasium` (inside `try/except ModuleNotFoundError`) at the
top of `jlab_opt_control/__init__.py` and above `import tensorflow` in the driver
script. The driver resolves `AGS-Injection-v0` like a built-in then — the `gym`
branch of `create_and_configure_env`. Undo: `git checkout` those two files.

### 2.5 Verify

```bash
cd SciOptControlToolkit                # any cwd works; logdirs/paths are self-located
python -m jlab_opt_control.drivers.run_continuous \
    --env AGS-Injection-v0 --nepisodes 2 --logdir /tmp/ags_smoke
```

Look for `Created gym environment: AGS-Injection-v0`, `Size of State Space -> 315`,
`Size of Action Space -> 96`, and `Training Episodic Reward` lines in the **random-policy
band ≈ −110…−130** (the lattice is re-drawn unseeded per process — see §5). First run
compiles the Julia stack (a few minutes); steady state ≲1 s/episode.

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
    --env AGS-Injection-v0 --agent KerasTD3-v0 --nepisodes 10000 \
    --inference_interval 50 --nepisode_avg 20 \
    --logdir results/ags_td3
```

(Requires the `__init__.py` block above — or run through the `ags-soct-run` fallback instead.)

Expect: first episode ~40 s (Julia first-compile inside the process), steady state ~0.3–1.5
s/episode → **10k episodes ≈ 1 hour** on CPU. First 2500 episodes (`warmup_size`) are pure random
exploration — the reward only starts moving after that.

Known-good reference: the 2026-10-08 run went **−117.6 → −107** (best −106.9 @ ep 9950). See §5
for what "good" means on this env — do not chase −0.

### 3.2 Common flags

| flag | default | note |
|---|---|---|
| `--env` | `Pendulum-v1` (SOCT default) | pass `--env AGS-Injection-v0`; other SOCT/gym envs still reachable |
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

There is **no env cfg file** for this environment. Knobs:

| knob | where | default | effect |
|---|---|---|---|
| `misalign_sigma` | `AGSEnv.__init__` (set via the launcher's `register(...)` kwargs, `soct_launcher.py`) | `3.4e-4` | quad misalignment σ (m); drawn **once at env construction** — the lattice is frozen for the whole run |
| `h_tune`, `v_tune` | same | `0.0` | fixed tune-control settings (not actions) |
| corrector bounds | `AGS_GymEnv.jl` `I_dhc/dvc_bounds` | ±25 | action box (A) — the binding performance limit |
| reward λ | `AGS_GymEnv.jl` `get_reward` | 1000 | scale on −λ·RMS offset |
| BPM noise | `AGS_GymEnv.jl` `_dS/_dX/_dY` | 0.0 | electrical noise (zero = deterministic) |
| beam params | `AGS_GymEnv.jl` `BEAM_*` consts | — | emittance, centroid, dispersion, population |

- **State (315)** = 73 BPMs × (S, Dx, Dy) sorted by BPM name + 48 `I_dhc` + 48 `I_dvc` currents.
  `S` = charge·alive_fraction·cal (10.014 = full survival); `Dx/Dy` = surviving-bunch centroid × S.
- **Action (96)**, box ±25: **block layout** — first 48 = sorted `dhc`, next 48 = sorted `dvc`
  (the Julia `set_action` mapping is authoritative).
- **Reward** = −1000 · RMS(signal-weighted orbit offset) — maximize, max 0. One caveat: since
  `Dx = x̄·S`, **total beam loss also gives reward 0**. Full-survival solutions bottom out near
  −105 ± a few (see §5), so anything drifting toward 0 is loss, not skill. `src/tracking.jl` is a
  standalone analysis script (main-magnet misalignments) — it is *not* part of the env path.

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

A healthy AGS run: flat ≈ −117 through ep 2500 (warmup), first gain ~3000, ≈ −109 by ~4000,
slow grind toward **−105…−108**. **This env is near-solved by construction:** the ±25 A corrector
authority bounds the best achievable reward at the box-constrained least-squares optimum,
measured ≈ −100…−105 (zero-corrector baseline ≈ −110…−118). Inference reward in that band with
all S ≈ 10.014 *is* success; big positive swings mean beam loss, not learning.

### Evaluating a saved policy

```bash
python ../AGS-sim-env/soct_launcher.py --inference --nepisodes 3 \
    --load_model <run>/models/epoch_10000 --logdir /tmp/eval
```

`AGS-sim-env/verify_policy.py` additionally loads every checkpoint dir you point it at and prints
per-checkpoint reward vs zero-action plus the min/max S survival check
(`python ../AGS-sim-env/verify_policy.py <run_dir>` from SOCT).

Note: **misalignments are re-drawn unseeded at env construction**, so every new process
(verification, `--inference` runs, retraining) evaluates a *different frozen lattice*. Expect
evaluation rewards to differ by a few units run-to-run — that is normal, not instability.

## 6. Notes

- TF runs CPU-only out of the box in this conda env (its CUDA wheels vs system driver mismatch);
  batch 256 nets are small, CPU ≈ fine. `Could not find cuda drivers` warnings are expected.
- Everything relative (run dirs, git hash) keys off the CWD you launch from — stay inside
  `SciOptControlToolkit/`.
- Episode clock note: `AGS_GymEnv.jl` re-measures the ring on every `reset()` and `step()`
  (single-pass sampling, ≲0.3 s each) — that dominates the episode time, not TF.
- If `import juliacall` ever fails with `sys.so ... cannot open`, Julia is not 1.12 (§2.2) or
  juliaup's cache changed — the launcher's pins (§2.5) override auto-detection.

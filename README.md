# AGS Injection Simulation Environment

Start julia inside the cloned repo's environment with `julia --project`

You may need to instantiate the environment inside julia with 
```
julia>  ]instantiate
```

## Python Gymnasium Environment

This package includes a Python gymnasium environment that interfaces with the
Julia beam-tracking simulation via [juliacall](https://github.com/JuliaPy/PythonCall.jl)
(the Python side of PythonCall.jl). All physics (machine, apertures, beam, tracking)
lives in `src/tracking.jl` — the single source of truth; `src/AGS_GymEnv.jl` is a thin
facade exposing only measurement/control primitives, and all RL logic (state assembly,
reward, action layout, knobs) lives in `ags_env.py`. To train an agent with
SciOptControlToolkit, use `soct_launcher.py` / the `ags-soct-run` console script — see
`run_guide.md` for the full quickstart.

### Installation

```bash
conda activate <your-ml-env>
pip install juliacall==0.9.35         # must match the Manifest's PythonCall version
pip install -e /path/to/AGS-sim-env   # MUST be editable: the Julia project
                                      # (Project.toml/Manifest.toml/src/) lives
                                      # in this checkout
```

Then, to use it from SciOptControlToolkit like a built-in
(`python -m jlab_opt_control.drivers.run_continuous --env AGS-Injection-v0`), run:

```bash
ags-install-soct        # idempotent; optional arg: path to the SOCT checkout
```

It inserts a guarded `import ags_gymnasium` at the **top** of
`jlab_opt_control/__init__.py` (above the TensorFlow-loading import chain) and
above `import tensorflow` in the driver — placement is load-bearing, because
embedding libjulia after TensorFlow's runtime segfaults the process. The blocks
no-op when AGS isn't installed; undo with `git checkout` of the two files.
Standalone scripts must likewise `import ags_gymnasium` before TensorFlow —
Gymnasium ≥ 1.0 removed plugin auto-loading, so the import is what registers the
id (the `gymnasium.env` entry points in `pyproject.toml` only serve
Gymnasium ≤ 0.29). Or skip everything: `ags-soct-run` needs neither the SOCT
edit nor an import.

### Usage (installed)

```python
import ags_gymnasium                # registers the env + starts Julia; before TF!
import gymnasium as gym

env = gym.make("AGS-Injection-v0")  # TimeLimit(1) wrapper, block-layout actions

state, info = env.reset()
print(f"State shape: {state.shape}")  # (315,)

action = env.action_space.sample()    # 96-dim, ±25 A
next_state, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}")            # ~6-27 (default = death lattice, see below)
print(f"Terminated: {terminated}")    # True (single-step optimization)
```

### Usage (direct class, from the repo)

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # if run from elsewhere
from soct_integration import configure_julia
configure_julia()  # pins Julia 1.12 for juliacall — before juliacall is imported

from ags_env import AGSEnv

# h_tune / v_tune fixed during episodes but configurable;
# misalign_sigma = MAIN-MAGNET misalignment sigma (m), drawn ONCE at
# construction over the six mm_* groups. Default 1e-2 == the src/tracking.jl
# load-time physics default: the beam dies at sector C (mean S ~0.06-0.27
# of full, 53-68/73 BPMs dead) and NOTHING at +-25 A recovers it (solvers,
# greedy, 100 random: all within +0.2 pt of baseline - study doc section 7).
# Rewards therefore sit ~6-27 with no learnable gradient. For a learnable
# task pass misalign_sigma=1.5e-3 (archived task point, study doc section 6).
env = AGSEnv(
    h_tune=0.0,
    v_tune=0.0,
)
```

`env.bpm_names` / `env.control_names` give the sorted names (73 / 96).

### Environment Specification

- **State space** (315 dimensions):
  - 73 BPM observables × 3 (S, Dx, Dy) = 219 values
  - 48 I_dhc current values
  - 48 I_dvc current values

- **Action space** (96 dimensions):
  - 48 I_dhc current values (bounded ±25 A)
  - 48 I_dvc current values (bounded ±25 A)

- **Objective / reward**: MAX OUT the BPM signal S (survived bunch intensity)
  along the whole line:
  `reward = 100 x mean(S/s_full) + 0.1 x (1 - min(RMS/rms_ref, 3))` in [0, 100].
  Primary term is survival (losing beam can never raise the reward); the
  small centering term only breaks full-survival plateau ties. `s_full` and
  `rms_ref` are measured on the constructed lattice. The legacy
  `-1000 x RMS(Dx,Dy)` centroid objective is superseded (it scored total
  beam loss ~0, above every live state); it remains computable from the
  state vector (Dx, Dy are observed).

- **Episode**: Single-step optimization — agent sets controls once, observes
  outcome, receives reward, and episode terminates.

### Testing

Run the test suite:

```bash
python test_env.py
```

## Julia Usage

To simulate one pass of a beam with 1000 macroparticles:
```
julia>  measure_observables!(
           beam,
           ags,
           pue,
           verbose = true
        )
```

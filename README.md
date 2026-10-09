# AGS Injection Simulation Environment

Start julia inside the cloned repo's environment with `julia --project`

You may need to instantiate the environment inside julia with 
```
julia>  ]instantiate
```

## Python Gymnasium Environment

This package includes a Python gymnasium environment that interfaces with the
Julia simulation via [juliacall](https://github.com/JuliaPy/PythonCall.jl)
(the Python side of PythonCall.jl). To train an agent with SciOptControlToolkit,
use `soct_launcher.py` / the `ags-soct-run` console script — see `run_guide.md`
for the full quickstart.

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
print(f"Reward: {reward}")            # ~-110 with zero correctors
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
# misalign_sigma = quadrupole misalignment σ, drawn ONCE at construction
env = AGSEnv(
    h_tune=0.0,
    v_tune=0.0,
    misalign_sigma=3.4e-4,
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

- **Reward**: −1000 × RMS signal-weighted orbit offset across all BPMs

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

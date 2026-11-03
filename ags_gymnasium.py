"""Gymnasium registration for the AGS injection environment.

Usage (after `pip install -e AGS-sim-env`):

    import ags_gymnasium            # registers AGS-Injection-v1 + starts Julia
    import gymnasium as gym
    env = gym.make("AGS-Injection-v1")

IMPORTANT: import this module BEFORE importing TensorFlow. It eagerly imports
juliacall (via ags_env); libjulia loaded after TensorFlow's runtime segfaults
(observed exit 139 with TF 2.21). For SOCT training runs, prefer
`soct_launcher.py` / `ags-soct-run`, which handle the import order for you.
"""
import sys

if "tensorflow" in sys.modules and "juliacall" not in sys.modules:
    raise RuntimeError(
        "ags_gymnasium: TensorFlow is already imported and Julia is not loaded — "
        "embedding libjulia after TF's runtime segfaults. Import ags_gymnasium "
        "before TensorFlow (SOCT does this in jlab_opt_control/__init__.py), or "
        "launch via `ags-soct-run`, which handles the ordering for you."
    )

import gymnasium as gym

from soct_integration import configure_julia

configure_julia()  # must precede juliacall import

import ags_env  # noqa: E402  (imports juliacall; must precede TensorFlow)

# v1: S-max survival reward + 3.4e-3 / +-50 A task point + normalized
# actions (study doc section 9, 2026-10-31). The pre-2026-10-21 ID
# AGS-Injection-v0 carried the legacy -1000*RMS centering reward; old v0
# run logs refer to that objective and are NOT comparable to v1 runs, so
# no v0 alias is registered.
if "AGS-Injection-v1" not in gym.envs.registry:
    gym.register(
        id="AGS-Injection-v1",
        entry_point=ags_env.AGSEnv,
        max_episode_steps=1,  # single-step optimization task
    )

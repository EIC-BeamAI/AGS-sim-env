#!/usr/bin/env python3
"""Launch a SOCT training/eval run against the AGS env defined in this repo.

Everything AGS-specific lives here or in ags_env.py; SciOptControlToolkit is
used unmodified. The launcher:
  1. pins juliacall's embedded Julia to the real 1.12 install (juliaup's
     launcher breaks bindir resolution under embedded init — override with
     PYTHON_JULIACALL_EXE/LIB exports if your layout differs),
  2. starts Julia directly inside this project (PYTHON_JULIACALL_PROJECT),
  3. imports juliacall BEFORE TensorFlow (see ags_env.py),
  4. registers AGS-Injection-v0 in SOCT's env registry,
  5. hands off to jlab_opt_control.drivers.run_continuous.main().

Run from inside the SciOptControlToolkit checkout so the driver records its
git hash and writes relative logdirs there:

    cd SciOptControlToolkit
    conda activate jlab_opt_control
    python ../AGS-sim-env/soct_launcher.py \
        --env AGS-Injection-v0 --agent KerasTD3-v0 --nepisodes 10000 \
        --inference_interval 50 --nepisode_avg 20 --logdir results/ags_td3
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from soct_integration import configure_julia, register_env

try:
    configure_julia()  # pins juliacall's Julia — must run before importing juliacall
except RuntimeError as e:
    sys.exit(str(e))

import juliacall  # noqa: E402  (must precede TensorFlow — see ags_env.py header)

import jlab_opt_control.envs as soct_envs  # noqa: E402

register_env(soct_envs)

from jlab_opt_control.drivers.run_continuous import main as run_driver  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--env" not in argv:  # AGS is this launcher's purpose; SOCT defaults to Pendulum
        argv = ["--env", "AGS-Injection-v0"] + argv
    return run_driver(argv)


if __name__ == "__main__":
    sys.exit(main())

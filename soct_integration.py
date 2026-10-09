"""Shared bootstrap for using the AGS env from Python.

Import order matters: call configure_julia() BEFORE importing juliacall
(or anything that imports it, e.g. ags_env), and import juliacall BEFORE
TensorFlow (libjulia vs TF threading segfault — see ags_env.py header).
"""
import glob
import os


def configure_julia():
    """Pin juliacall's embedded Julia to a real 1.12 install and this project.

    juliaup's launcher binary breaks bindir resolution under embedded init,
    so point PYTHON_JULIACALL_EXE/LIB at the newest juliaup 1.12 install and
    start Julia directly inside this directory's project (Manifest.toml).
    Pre-set PYTHON_JULIACALL_EXE/LIB/PROJECT to override.
    """
    if "PYTHON_JULIACALL_EXE" not in os.environ:
        cands = sorted(glob.glob(os.path.expanduser("~/.julia/juliaup/julia-1.12*")))
        if not cands:
            raise RuntimeError(
                "soct_integration: no Julia 1.12 under ~/.julia/juliaup.\n"
                "Install/activate it (`juliaup default 1`) or export "
                "PYTHON_JULIACALL_EXE and PYTHON_JULIACALL_LIB yourself."
            )
        jlhome = cands[-1]
        os.environ["PYTHON_JULIACALL_EXE"] = os.path.join(jlhome, "bin", "julia")
        libs = sorted(glob.glob(os.path.join(jlhome, "lib", "libjulia.so.1.12*")))
        if libs:
            os.environ["PYTHON_JULIACALL_LIB"] = libs[-1]
    project = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(project, "Project.toml")):
        raise RuntimeError(
            "soct_integration: the Julia project (Project.toml) is not next to "
            "this module — the package was installed non-editable. Reinstall with "
            "`pip install -e <AGS-sim-env>` or point PYTHON_JULIACALL_PROJECT at "
            "the AGS-sim-env checkout."
        )
    os.environ.setdefault("PYTHON_JULIACALL_PROJECT", project)


def register_env(soct_envs):
    """Register AGS-Injection-v0 in SOCT's env registry (idempotent).

    entry_point resolves via sys.path — callers must have inserted this
    directory first so `ags_env` is importable as a top-level module.
    """
    if "AGS-Injection-v0" not in soct_envs.list_registered_modules():
        soct_envs.register(
            id="AGS-Injection-v0",
            entry_point="ags_env:AGSEnv",
            kwargs={"max_episode_steps": 1},
        )

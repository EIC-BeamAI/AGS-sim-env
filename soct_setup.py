"""One-shot SOCT setup helper: registers AGS-Injection-v0 in a SciOptControlToolkit checkout.

Run this AFTER `pip install -e <AGS-sim-env>` in the Python environment that
runs SOCT:

    ags-install-soct [PATH_TO_SCIOPTCONTROLTOOLKIT]

(If the console script isn't on PATH: `python soct_setup.py` from this repo.)

It makes exactly two edits, both guarded so they no-op when AGS isn't
installed, and both idempotent (a second run changes nothing):

  1. top of  jlab_opt_control/__init__.py            — import ags_gymnasium
  2. in     jlab_opt_control/drivers/run_continuous.py, above
           `import tensorflow`                        — same import, for
           direct-script mode (`python .../run_continuous.py`)

Why the placement: embedding libjulia AFTER TensorFlow's runtime segfaults the
process, and `import jlab_opt_control` pulls TF in during its own __init__ —
so the registration must be the first thing the package does. envs/__init__.py
is too late; don't move the block there.

Undo: `git checkout jlab_opt_control/__init__.py jlab_opt_control/drivers/run_continuous.py`
inside the SOCT repo.
"""
import argparse
import importlib.util
import os
import py_compile
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_INIT_MARKER = "# --- AGS-Injection-v0 registration (added by ags-install-soct) ---"
_DRV_MARKER = "# --- AGS-Injection-v0 import (added by ags-install-soct) ---"

_INIT_BLOCK = """{marker}
# Registers AGS-Injection-v0 in the gymnasium registry. Must run before
# TensorFlow is imported: embedding libjulia after TF's runtime segfaults the
# process. Silently skipped when the AGS package (`pip install -e AGS-sim-env`)
# is not installed in this environment.
try:
    import ags_gymnasium  # noqa: F401
except ModuleNotFoundError:
    pass
# --- end AGS ---
""".format(marker=_INIT_MARKER)

_DRV_BLOCK = """{marker}
# Registers AGS-Injection-v0 (must precede tensorflow; see
# jlab_opt_control/__init__.py). No-op for `python -m` runs — the package
# __init__ already did it. Silently skipped when AGS is not installed.
try:
    import ags_gymnasium  # noqa: F401
except ModuleNotFoundError:
    pass
# --- end AGS ---
""".format(marker=_DRV_MARKER)


def find_soct_root(cli_path=None):
    """Locate the jlab_opt_control package directory."""
    if cli_path:
        pkg = os.path.join(os.path.abspath(cli_path), "jlab_opt_control")
        if not os.path.isdir(pkg):
            cli_path = os.path.join(os.path.abspath(cli_path), "jlab_opt_control")
            pkg = cli_path if os.path.isdir(cli_path) else None
    else:
        spec = importlib.util.find_spec("jlab_opt_control")
        pkg = os.path.dirname(spec.origin) if spec and spec.origin else None
        if pkg is None:
            sibling = os.path.join(os.path.dirname(HERE), "SciOptControlToolkit",
                                   "jlab_opt_control")
            pkg = sibling if os.path.isdir(sibling) else None
    if pkg is None or not os.path.isfile(os.path.join(pkg, "__init__.py")):
        sys.exit(
            "ags-install-soct: could not locate a jlab_opt_control package.\n"
            "Pass the SciOptControlToolkit checkout explicitly:\n"
            "  ags-install-soct /path/to/SciOptControlToolkit"
        )
    return pkg


def _write(path, marker, src, new_src):
    if marker in src or "import ags_gymnasium" in src:
        print(f"  already registered: {path}")
        return False
    with open(path, "w") as f:
        f.write(new_src)
    py_compile.compile(path, doraise=True)
    print(f"  patched: {path}")
    return True


def patch_init(pkg_dir):
    path = os.path.join(pkg_dir, "__init__.py")
    with open(path) as f:
        src = f.read()
    return _write(path, _INIT_MARKER, src, _INIT_BLOCK + "\n" + src)


def patch_driver(pkg_dir):
    path = os.path.join(pkg_dir, "drivers", "run_continuous.py")
    if not os.path.isfile(path):
        print(f"  skipped (file not found): {path}")
        return False
    with open(path) as f:
        src = f.read()
    m = re.search(r"^(?:import tensorflow|from tensorflow)[^\n]*$", src, re.M)
    if m is None:
        m = re.search(r"^(?:import|from) \S+", src, re.M)
    pos = m.start() if m else 0
    return _write(path, _DRV_MARKER, src,
                  src[:pos] + _DRV_BLOCK + "\n" + src[pos:])


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ags-install-soct",
        description="Register AGS-Injection-v0 in a SciOptControlToolkit checkout.")
    ap.add_argument("soct_path", nargs="?", default=None,
                    help="path to the SciOptControlToolkit checkout "
                         "(default: auto-detect via the active Python environment)")
    args = ap.parse_args(argv)

    pkg = find_soct_root(args.soct_path)
    print(f"Target: {pkg}")
    changed = [patch_init(pkg), patch_driver(pkg)]

    if any(changed):
        print("\nDone. Verify with:")
        print("  cd <SciOptControlToolkit>")
        print("  python -m jlab_opt_control.drivers.run_continuous "
              "--env AGS-Injection-v0 --nepisodes 2 --logdir /tmp/ags_smoke")
        print("expect: 'Created gym environment: AGS-Injection-v0', 315/96, reward ~-110...-120")
    else:
        print("\nNothing to do — SOCT is already set up for AGS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

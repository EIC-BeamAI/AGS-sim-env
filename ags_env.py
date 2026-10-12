"""Python-side Gymnasium wrapper for the AGS injection simulation.

Architecture: ALL physics (machine, apertures, beam, tracking) lives in
Julia's `src/tracking.jl` — the single source of truth, kept identical to the
physics branch. `src/AGS_GymEnv.jl` is a thin facade exposing only marshalling
primitives (redraw misalignments, set/zero corrector currents, set tunes, one
ring measurement). Everything RL-specific — the 315-dim state assembly, reward,
action layout and bounds, and all env configuration — is THIS file, so physics
updates in tracking.jl land here without any Julia edits.

Import juliacall BEFORE numpy/tensorflow: Julia's libjulia conflicts with
TensorFlow's runtime when TF is loaded first (segfault ordering, see README).
"""
import os

import juliacall

jl = juliacall.Main

# Activate the AGS-sim-env Julia project (this file lives in the project root)
_ags_sim_env = os.path.dirname(os.path.abspath(__file__))
jl.seval(f'import Pkg; Pkg.activate("{_ags_sim_env}")')
jl.seval('using AGS_GymEnv')

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_AGS = jl.AGS_GymEnv
_REDRAW = getattr(_AGS, "redraw_misalignments!")
_SET_CURRENTS = getattr(_AGS, "set_currents!")
_ZERO_CURRENTS = getattr(_AGS, "zero_currents!")
_SET_TUNES = getattr(_AGS, "set_tunes!")
_MEASURE = _AGS.measure
_SYM = {k: jl.Symbol(k) for k in ("S", "Dx", "Dy")}

# =====================================================================
# RL configuration (Python owns these; see run_guide.md §4)
# =====================================================================

#: Reward = -LAMBDA * RMS(signal-weighted orbit offset). Maximize; 0 is best.
LAMBDA_REWARD = 1000.0

#: Dipole-corrector current bounds (A) — the binding performance limit.
ACTION_BOUND = 25.0

#: Default main-magnet misalignment sigma (m), matching the mm_* draws in
#: tracking.jl. Drawn once per env construction (frozen lattice per process).
DEFAULT_MISALIGN_SIGMA = 1e-2

#: BPM electrical noise (RMS) on S, Dx, Dy; zero keeps the env deterministic.
DEFAULT_BPM_NOISE = (0.0, 0.0, 0.0)


class AGSEnv(gym.Env):
    """
    Gymnasium environment for AGS injection using the Julia tracking kernel.

    The agent sets 96 dipole-corrector currents to center the beam at all 73
    BPMs. Reward = -1000 * RMS(Dx^2 + Dy^2 over BPMs); a fully centered,
    fully surviving beam scores 0 (total beam loss ALSO scores ~0 — read S!).

    State (315): 73 BPMs x (S, Dx, Dy) interleaved, sorted by BPM name,
                 then 48 sorted I_dhc + 48 sorted I_dvc currents.
    Action (96): block layout — 48 sorted I_dhc, then 48 sorted I_dvc, ±25 A.
    Episode: single-step optimization (terminated=True after one step).

    Physics (lattice, apertures, beam, tracking) = src/tracking.jl via the
    src/AGS_GymEnv.jl facade; this class = RL glue only.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        h_tune: float = 0.0,
        v_tune: float = 0.0,
        misalign_sigma: float = DEFAULT_MISALIGN_SIGMA,
        bpm_noise=DEFAULT_BPM_NOISE,
        render_mode=None,
        **kwargs
    ):
        """
        Args:
            h_tune: Horizontal tune control setting (fixed during episodes)
            v_tune: Vertical tune control setting (fixed during episodes)
            misalign_sigma: Main-magnet misalignment sigma (m); x_offset ~
                N(0, sigma) drawn ONCE here for all six mm_* groups — the
                lattice is frozen for the env's life (fresh draw per process).
            bpm_noise: (dS, dX, dY) electrical RMS noise; default zeros.
            render_mode: Unused.
        """
        self.h_tune = float(h_tune)
        self.v_tune = float(v_tune)
        self.misalign_sigma = float(misalign_sigma)
        self.bpm_noise = tuple(float(x) for x in bpm_noise)

        # Canonical orders (sorted names) — the state/action contract
        self.bpm_names = sorted(str(n) for n in _AGS.bpm_names())
        self._dhc_names = sorted(str(n) for n in _AGS.dhc_names())
        self._dvc_names = sorted(str(n) for n in _AGS.dvc_names())
        self.n_bpm = len(self.bpm_names)
        self.n_dhc = len(self._dhc_names)
        self.n_dvc = len(self._dvc_names)
        self.state_size = 3 * self.n_bpm + self.n_dhc + self.n_dvc
        self.action_size = self.n_dhc + self.n_dvc
        self.control_names = [
            f"{base}_{kind}" for base in self._dhc_names for kind in ("dhc", "dvc")
        ]

        # Configure the machine (frozen lattice for this process)
        _SET_TUNES(self.h_tune, self.v_tune)
        _REDRAW(self.misalign_sigma)
        _ZERO_CURRENTS()
        self._dhc = np.zeros(self.n_dhc, dtype=np.float64)
        self._dvc = np.zeros(self.n_dvc, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=np.full(self.state_size, -1000.0, dtype=np.float64),
            high=np.full(self.state_size, 1000.0, dtype=np.float64),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-ACTION_BOUND,
            high=ACTION_BOUND,
            shape=(self.action_size,),
            dtype=np.float64,
        )

        self.render_mode = render_mode

    # ------------------------------------------------------------------
    # Julia/observables marshalling
    # ------------------------------------------------------------------

    def _measure(self):
        """One single-pass ring measurement -> (S, Dx, Dy) arrays in BPM order."""
        dS, dX, dY = self.bpm_noise
        obs = _MEASURE(dS=dS, dX=dX, dY=dY)
        S = np.empty(self.n_bpm, dtype=np.float64)
        Dx = np.empty(self.n_bpm, dtype=np.float64)
        Dy = np.empty(self.n_bpm, dtype=np.float64)
        for i, name in enumerate(self.bpm_names):
            bpm = obs[name]
            S[i] = bpm[_SYM["S"]]
            Dx[i] = bpm[_SYM["Dx"]]
            Dy[i] = bpm[_SYM["Dy"]]
        return S, Dx, Dy

    @staticmethod
    def _reward(Dx, Dy):
        """-LAMBDA * RMS(signal-weighted orbit offset) — matches tracking.jl's
        Dx = x_centroid * S convention; total loss also scores ~0 (check S)."""
        rms = np.sqrt(np.mean(Dx * Dx + Dy * Dy))
        return -LAMBDA_REWARD * rms

    def _state(self, S, Dx, Dy):
        """315-dim vector: [S,Dx,Dy] interleaved per BPM (sorted names), then
        dhc (sorted), then dvc (sorted) — layout unchanged across the refactor,
        so checkpoints stay valid."""
        bpm_block = np.column_stack((S, Dx, Dy)).ravel()
        return np.concatenate((bpm_block, self._dhc, self._dvc))

    def _run(self):
        """Measure, assemble state, score reward — the whole per-step pipeline."""
        S, Dx, Dy = self._measure()
        return self._state(S, Dx, Dy), self._reward(Dx, Dy)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Zero all corrector currents and re-measure. Misalignments are NOT
        redrawn — the lattice is fixed for the env's lifetime (set in __init__).

        Returns:
            state: Initial state vector (zero currents)
            info: {"reward": zero-action reward}
        """
        super().reset(seed=seed)

        _ZERO_CURRENTS()
        self._dhc[:] = 0.0
        self._dvc[:] = 0.0

        state, reward = self._run()
        return state, {"reward": float(reward)}

    def step(self, action):
        """
        Apply corrector currents and measure once (single-step task).

        Args:
            action: 96-dim, block layout: n_dhc sorted dhc then n_dvc sorted
                dvc currents, within +/-25 A (clipped defensively).

        Returns:
            state, reward, terminated=True, truncated=False, {}
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(
                f"expected action shape ({self.action_size},), got {action.shape}"
            )
        action = np.clip(action, -ACTION_BOUND, ACTION_BOUND)

        self._dhc = action[: self.n_dhc].copy()
        self._dvc = action[self.n_dhc :].copy()
        _SET_CURRENTS(self._dhc_names, self._dhc, self._dvc_names, self._dvc)

        state, reward = self._run()
        return state, float(reward), True, False, {}

    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode == "human":
            print(f"State size: {self.state_size}")
            print(f"Action size: {self.action_size}")

    def close(self):
        """Clean up environment resources."""
        pass

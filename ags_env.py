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
# RL configuration (Python owns these; see docs/run_guide.md §4)
# =====================================================================

#: TASK OBJECTIVE: max out BPM signal S (survived bunch intensity) along the
#: whole line.  reward = 100*mean(S/S_full) + SHAPE_W*(1 - min(RMS/RMS_ref, 3))
#: in [0, 100]. Primary term IS the objective: losing beam can never raise
#: the reward. The tie-breaker (the legacy centroid objective, down-weighted)
#: breaks full-survival plateau ties and supplies gradient where dS/dI == 0
#: (S is a staircase: it moves only when macroparticles clip apertures).
#: S_full and RMS_ref are measured on THIS lattice at construction (the RMS
#: convention here is mean-over-concat; do not mix with the old Julia
#: /n_bpm = sqrt(2)x convention from study runs before 2026-10-21).
SHAPE_W = 0.1
SHAPE_CLIP = 3.0

#: Dipole-corrector current bounds (A). Task point 2026-10-31 (study doc
#: env_research/env_design_study.md section 9): +-50 A is the learning
#: optimum -- both TD3 arms reach 99.3-99.8 median there and are jitter-
#: robust; 25-100 A workable, 150-300 A degraded, >=500 A collapses for
#: from-scratch training. Greedy ceiling ~0.99 exists at ALL bounds >=50 A,
#: so larger boxes only make the task harder to LEARN, never more solvable.
#: Override with env var AGS_ACTION_BOUND (study protocol: 50 at peak).
ACTION_BOUND = float(os.environ.get("AGS_ACTION_BOUND", "50.0"))

#: Default main-magnet misalignment sigma (m) -- task point 2026-10-31.
#: 3.4e-3 gives a hard-but-learnable lattice: zero action ~40 on the
#: default lattice 20261022 (~55 median over fresh draws); every BPM is
#: alive so the LS optimum (~51-54%) is informative; RL reaches 99+ at
#: +-50 A. The 1e-2 physics default kills the beam at
#: sector C (mean S ~0.18, 60/73 BPMs dead, LS meaningless); the 1.5e-3
#: archived point is trivially survivable (~99% uncorrected). See study
#: doc sections 6-9. Override with env var AGS_SIGMA_TASK.
DEFAULT_MISALIGN_SIGMA = float(os.environ.get("AGS_SIGMA_TASK", "3.4e-3"))

#: BPM electrical noise (RMS) on S, Dx, Dy; zero keeps the env deterministic.
DEFAULT_BPM_NOISE = (0.0, 0.0, 0.0)


class AGSEnv(gym.Env):
    """
    Gymnasium environment for AGS injection using the Julia tracking kernel.

    The agent sets 96 dipole-corrector currents to MAX OUT the BPM signal S
    (survived bunch intensity) along the whole line:
    reward = 100*mean(S/s_full) + 0.1*(1 - min(RMS/rms_ref, 3)) in [0, 100].
    Losing beam can never raise the reward; the centering tie-breaker only
    orders full-survival plateau configs and supplies gradient where the
    (staircase) survival is locally flat. s_full and rms_ref are measured on
    the constructed lattice at __init__.

    State (315): 73 BPMs x (S, Dx, Dy) interleaved, sorted by BPM name,
                 then 48 sorted I_dhc + 48 sorted I_dvc currents.
    Action (96): block layout — 48 sorted I_dhc, then 48 sorted I_dvc.
    Default normalized [-1, 1] scaled by ACTION_BOUND inside step()
    (norm_action=True); norm_action=False gives raw amp box +-ACTION_BOUND.
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
        norm_action: bool = True,
        lattice_seed: int = 20261022,
        render_mode=None,
        **kwargs
    ):
        """
        Args:
            h_tune: Horizontal tune control setting (fixed during episodes)
            v_tune: Vertical tune control setting (fixed during episodes)
            misalign_sigma: Main-magnet misalignment sigma (m); x_offset ~
                N(0, sigma) drawn ONCE here for all six mm_* groups — the
                lattice is frozen for the env's life. Draws come from the
                Julia global RNG seeded to lattice_seed, so the same seed
                reproduces the same lattice (benchmark continuity).
            bpm_noise: (dS, dX, dY) electrical RMS noise; default zeros.
            norm_action: True (default) -> action space [-1, 1], scaled by
                ACTION_BOUND inside step() (R8: keeps exploration/smoothing
                in action units independent of the physical bound; TD3
                smoothing sigma 0.2 = 10 A at 50 A bound, but 300 A —
                lethal — at 1500 A). False -> raw ampere box.
            lattice_seed: Julia RNG seed before the lattice draws.
            render_mode: Unused.
        """
        self.h_tune = float(h_tune)
        self.v_tune = float(v_tune)
        self.misalign_sigma = float(misalign_sigma)
        self.bpm_noise = tuple(float(x) for x in bpm_noise)
        self.norm_action = bool(norm_action)
        self.lattice_seed = int(lattice_seed)
        self.action_bound = ACTION_BOUND
        # Main.Random is not bound here (tracking.jl only `import Random:
        # default_rng`); import into Main via Base.eval because seval wraps
        # only the final expression and would lose a `using` in an earlier one.
        jl.seval('Base.eval(Main, :(import Random))')
        jl.seval(f"Base.eval(Main, :(Random.seed!({self.lattice_seed})))")

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
        self._dhc = np.zeros(self.n_dhc, dtype=np.float64)
        self._dvc = np.zeros(self.n_dvc, dtype=np.float64)

        # Reward reference constants, measured on this machine at build time:
        #   s_full  = full-survival BPM signal (zero-misalignment pass)
        #   rms_ref = zero-current orbit RMS on the task lattice (shaper
        #             denominator, mean-over-concat convention)
        _REDRAW(0.0)
        _ZERO_CURRENTS()
        self.s_full = float(np.max(self._measure()[0]))
        _REDRAW(self.misalign_sigma)
        _ZERO_CURRENTS()
        _, Dx0, Dy0 = self._measure()
        self.rms_ref = max(float(np.sqrt(np.mean(Dx0 * Dx0 + Dy0 * Dy0))), 1e-12)

        self.observation_space = spaces.Box(
            low=np.full(self.state_size, -1000.0, dtype=np.float64),
            high=np.full(self.state_size, 1000.0, dtype=np.float64),
            dtype=np.float64,
        )
        if self.norm_action:
            # normalized action units; step() multiplies by ACTION_BOUND
            lo, hi = -1.0, 1.0
        else:
            lo, hi = -self.action_bound, self.action_bound
        self.action_space = spaces.Box(
            low=lo,
            high=hi,
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

    def _reward(self, S, Dx, Dy):
        """Task objective: MAX OUT BPM signal S along the whole line.
        Primary = 100*mean(clip(S/s_full)); a small down-weighted centering
        RMS term breaks full-survival plateau ties (never trades beam for
        cosmetics: quantum of one particle = 100/s_full/n_bpm of primary)."""
        rel = np.clip(S / self.s_full, 0.0, 1.0)
        rms = float(np.sqrt(np.mean(Dx * Dx + Dy * Dy)))
        return 100.0 * float(rel.mean()) + SHAPE_W * (
            1.0 - min(rms / self.rms_ref, SHAPE_CLIP))

    def _state(self, S, Dx, Dy):
        """315-dim vector: [S,Dx,Dy] interleaved per BPM (sorted names), then
        dhc (sorted), then dvc (sorted) — layout unchanged across the refactor,
        so checkpoints stay valid."""
        bpm_block = np.column_stack((S, Dx, Dy)).ravel()
        return np.concatenate((bpm_block, self._dhc, self._dvc))

    def _run(self):
        """Measure, assemble state, score reward — the whole per-step pipeline."""
        S, Dx, Dy = self._measure()
        return self._state(S, Dx, Dy), self._reward(S, Dx, Dy)

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
                dvc. norm_action: units [-1, 1] (clipped defensively),
                scaled by ACTION_BOUND here. norm_action=False: amperes,
                clipped to +-ACTION_BOUND.

        Returns:
            state, reward, terminated=True, truncated=False, {}
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(
                f"expected action shape ({self.action_size},), got {action.shape}"
            )
        if self.norm_action:
            action = np.clip(action, -1.0, 1.0) * self.action_bound
        else:
            action = np.clip(action, -self.action_bound, self.action_bound)

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

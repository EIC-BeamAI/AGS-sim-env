import sys
import os

# Import juliacall BEFORE numpy/tensorflow to avoid segfault
# This is critical - Julia's libjulia conflicts with TensorFlow's threading
# when TensorFlow is loaded first
import juliacall
jl = juliacall.Main

# Activate the AGS-sim-env Julia project (this file lives in the project root)
_ags_sim_env = os.path.dirname(os.path.abspath(__file__))
jl.seval(f'import Pkg; Pkg.activate("{_ags_sim_env}")')
jl.seval('using AGS_GymEnv')

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AGSEnv(gym.Env):
    """
    Gymnasium environment for AGS injection simulation using Julia backend.

    This environment wraps the Julia-based AGS beam tracking simulation.
    The agent controls dipole corrector currents to center the beam at all
    BPMs (reward = -1000 * RMS signal-weighted orbit offset; see
    src/AGS_GymEnv.jl get_reward).

    State space: 315 dimensions (73 BPMs x 3 + 48 I_dhc + 48 I_dvc),
        sorted by BPM/control name.
    Action space: 96 dimensions, block layout: 48 sorted I_dhc then 48
        sorted I_dvc currents; bounds read from AGS_GymEnv
        (currently +/-25 A via I_dhc_bounds/I_dvc_bounds).
    Episode: Single-step optimization (done after one step).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        h_tune: float = 0.0,
        v_tune: float = 0.0,
        misalign_sigma: float = 3.4e-4,
        render_mode=None,
        **kwargs
    ):
        """
        Initialize the AGS Gymnasium environment.

        Args:
            h_tune: Horizontal tune control setting (fixed during episodes)
            v_tune: Vertical tune control setting (fixed during episodes)
            misalign_sigma: Standard deviation for quadrupole misalignments
                (drawn ONCE here — the lattice is frozen for the env's life)
            render_mode: Rendering mode (not currently used)
        """
        # AGS_GymEnv module already loaded at import time
        self._julia_env = jl.seval('AGS_GymEnv')
        self._juliacall = juliacall

        # Store configuration
        self.h_tune = h_tune
        self.v_tune = v_tune
        self.misalign_sigma = misalign_sigma

        # Get environment info from Julia
        self.state_size = self._julia_env.get_state_size()
        self.action_size = self._julia_env.get_action_size()
        self.bpm_names = list(self._julia_env.get_bpm_names())
        self.control_names = list(self._julia_env.get_control_names())

        # Initialize the Julia environment with config
        getattr(self._julia_env, "initialize!")(
            h_tune=h_tune,
            v_tune=v_tune,
            misalign_sigma=misalign_sigma,
        )

        # Get action bounds from Julia
        low, high = self._julia_env.get_action_bounds()
        self._action_low = np.array(low, dtype=np.float64)
        self._action_high = np.array(high, dtype=np.float64)

        # Define observation and action spaces
        obs_low = np.full(self.state_size, -1000.0, dtype=np.float64)
        obs_high = np.full(self.state_size, 1000.0, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            dtype=np.float64,
        )

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Zeroes all corrector currents. Misalignments are NOT redrawn —
        they are fixed for the env's lifetime (set in __init__).

        Returns:
            state: Initial state vector
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        state, reward = self._julia_env.reset()
        state = np.array(state, dtype=np.float64)

        return state, {"reward": float(reward)}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Control currents to apply (96-dim array, block layout:
                48 sorted dhc then 48 sorted dvc)

        Returns:
            state: Next state vector
            reward: Reward value
            done: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Dictionary with additional information
        """
        action = np.asarray(action, dtype=np.float64)

        # Convert numpy array to Julia Vector{Float64} using pyconvert
        convert_py = self._juliacall.Main.seval('arr -> pyconvert(Vector{Float64}, arr)')
        action_jl = convert_py(action)

        state, reward, done = self._julia_env.step(action_jl)
        state = np.array(state, dtype=np.float64)

        return state, float(reward), bool(done), False, {}

    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode == "human":
            print(f"State size: {self.state_size}")
            print(f"Action size: {self.action_size}")

    def close(self):
        """Clean up environment resources."""
        pass

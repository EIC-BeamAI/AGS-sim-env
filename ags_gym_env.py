"""
AGS Gymnasium Environment for Python.

This module provides a gymnasium-compatible environment for the AGS injection
simulation. It interfaces with the Julia AGS_GymEnv package via PythonCall.jl.

State space:
    - 73 BPM observables × 3 (S, Dx, Dy) = 219 values
    - 48 I_dhc current values = 48 values
    - 48 I_dvc current values = 48 values
    Total: 315 dimensions

Action space:
    - 48 I_dhc current values
    - 48 I_dvc current values
    Total: 96 dimensions (Box(-10, 10, shape=(96,)))

Reward:
    Sum of S values across all BPMs (maximized)

Episode:
    Single-step optimization (always done after first step)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AGS_GymEnv(gym.Env):
    """
    Gymnasium environment for AGS injection simulation.

    This environment wraps the Julia-based AGS beam tracking simulation.
    The agent controls dipole corrector currents to maximize the beam
    signal at all BPMs.

    Attributes:
        state_size (int): Dimension of the state space (315)
        action_size (int): Dimension of the action space (96)
        bpm_names (list): List of BPM element names
        control_names (list): List of control variable names
        h_tune (float): Horizontal tune control (fixed)
        v_tune (float): Vertical tune control (fixed)
        misalign_sigma (float): Misalignment sigma for quadrupoles
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        h_tune: float = 0.0,
        v_tune: float = 0.0,
        misalign_sigma: float = 3.4e-3,
        render_mode=None,
    ):
        """
        Initialize the AGS Gymnasium environment.

        Args:
            h_tune: Horizontal tune control setting (fixed during episodes)
            v_tune: Vertical tune control setting (fixed during episodes)
            misalign_sigma: Standard deviation for quadrupole misalignments
            render_mode: Rendering mode (not currently used)
        """
        # Import Julia environment via PythonCall
        self._init_julia_env(h_tune, v_tune, misalign_sigma)

        # Define observation and action spaces
        self._setup_spaces()

        self.render_mode = render_mode

    def _init_julia_env(self, h_tune, v_tune, misalign_sigma):
        """Initialize the Julia environment using juliacall (PythonCall.jl)."""
        import juliacall
        import os

        # Determine the Julia project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize Julia with the correct project and load AGS_GymEnv
        # The project directory contains the AGS_GymEnv package, so we just
        # activate it and use the package directly
        init_code = f'''
            using Pkg
            Pkg.activate("{project_dir}")
            using AGS_GymEnv
            AGS_GymEnv
            '''
        self._julia_env = juliacall.Main.seval(init_code)
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
        # Use getattr for initialize! since ! is not valid in Python
        getattr(self._julia_env, "initialize!")(
            h_tune=h_tune,
            v_tune=v_tune,
            misalign_sigma=misalign_sigma,
        )

        # Get action bounds from Julia
        low, high = self._julia_env.get_action_bounds()
        self._action_low = np.array(low, dtype=np.float64)
        self._action_high = np.array(high, dtype=np.float64)

    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        # Observation space: state vector of size 315
        # BPM observables can range from -inf to +inf, but in practice
        # S values are around 10, Dx/Dy around ±0.2
        # Control currents range from -10 to 10
        # Use a generous range for the observation space
        obs_low = np.full(self.state_size, -1000.0, dtype=np.float64)
        obs_high = np.full(self.state_size, 1000.0, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float64,
        )

        # Action space: control currents bounded to [-10, 10]
        self.action_space = spaces.Box(
            low=self._action_low,
            high=self._action_high,
            dtype=np.float64,
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

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
            action: Control currents to apply (96-dim array)

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

    def get_bpm_names(self):
        """Get the list of BPM element names."""
        return self.bpm_names

    def get_control_names(self):
        """Get the list of control variable names."""
        return self.control_names

    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode == "human":
            print(f"State size: {self.state_size}")
            print(f"Action size: {self.action_size}")
        pass

    def close(self):
        """Clean up environment resources."""
        pass


def make_env(**kwargs):
    """
    Factory function to create an AGS gymnasium environment.

    Args:
        **kwargs: Arguments passed to AGS_GymEnv constructor

    Returns:
        AGS_GymEnv instance
    """
    return AGS_GymEnv(**kwargs)


if __name__ == "__main__":
    # Quick test
    env = AGS_GymEnv()
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    print(f"BPMs: {len(env.bpm_names)}")
    print(f"Controls: {len(env.control_names)}")

    state, _ = env.reset()
    print(f"Initial state shape: {state.shape}")
    old_state = state
    print(f"dS State before action: {state[::3]}")
    print(f"dX State before action: {state[1::3]}")

    # Test a zero action
    action = np.zeros(env.action_size)
    state, reward, done, truncated, info = env.step(action)
    print(f"dS State after action: {state[::3]}")
    print(f"dX State after action: {state[1::3]}")
    print(f"Change in state:  {abs(old_state-state)}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")

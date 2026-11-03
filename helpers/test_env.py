"""
Test script for the AGS Gymnasium environment.

Run with: python test_env.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.soct_integration import configure_julia

configure_julia()  # must run before importing juliacall (pulled in by ags_env)

import numpy as np
import gymnasium as gym


def test_env_construction():
    """Test that the environment can be constructed."""
    from ags_env import AGSEnv

    env = AGSEnv()
    print("✓ Environment constructed successfully")
    return env


def test_spaces(env):
    """Test that the observation and action spaces are correct."""
    assert env.state_size == 315, f"Expected state_size 315, got {env.state_size}"
    assert env.action_size == 96, f"Expected action_size 96, got {env.action_size}"
    assert env.observation_space.shape == (315,), \
        f"Expected observation shape (315,), got {env.observation_space.shape}"
    assert env.action_space.shape == (96,), \
        f"Expected action shape (96,), got {env.action_space.shape}"
    print(f"✓ State size: {env.state_size}")
    print(f"✓ Action size: {env.action_size}")
    print(f"✓ Observation space: {env.observation_space.shape}")
    print(f"✓ Action space: {env.action_space.shape}")


def test_bpm_and_control_names(env):
    """Test that BPM and control names are correctly retrieved."""
    n_bpm = len(env.bpm_names)
    n_controls = len(env.control_names)
    print(f"✓ BPM names: {n_bpm} BPMs")
    print(f"  First 5 BPMs: {env.bpm_names[:5]}")
    print(f"✓ Control names: {n_controls} controls")
    print(f"  First 5 controls: {env.control_names[:5]}")
    assert n_bpm == 73, f"Expected 73 BPMs, got {n_bpm}"
    assert n_controls == 96, f"Expected 96 controls, got {n_controls}"


def test_reset(env):
    """Test that reset returns a valid state."""
    state, info = env.reset()
    assert state.shape == (315,), f"Expected state shape (315,), got {state.shape}"
    assert isinstance(state, np.ndarray), f"Expected np.ndarray, got {type(state)}"
    assert not np.any(np.isnan(state)), "State contains NaN values"
    assert not np.any(np.isinf(state)), "State contains Inf values"
    print(f"✓ Reset succeeded, state shape: {state.shape}")
    return state


def test_step(env):
    """Test that step returns valid values."""
    # First reset
    state, _ = env.reset()

    # Test with zero action
    action = np.zeros(env.action_size)
    next_state, reward, done, truncated, info = env.step(action)
    assert next_state.shape == (315,), \
        f"Expected next_state shape (315,), got {next_state.shape}"
    assert isinstance(reward, float), f"Expected reward to be float, got {type(reward)}"
    assert done == True, "Expected done=True for single-step optimization"
    assert truncated == False, "Expected truncated=False"
    print(f"✓ Step with zero action succeeded")
    print(f"  Reward: {reward:.2f}")
    print(f"  Done: {done}")
    print(f"  Truncated: {truncated}")
    print(f"  Next state shape: {next_state.shape}")


def test_random_action(env):
    """Test that step works with a random action within bounds."""
    state, _ = env.reset()
    # Sample a random action within bounds
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    assert done == True, "Expected done=True for single-step optimization"
    print(f"✓ Step with random action succeeded")
    print(f"  Reward: {reward:.2f}")


def test_action_bounds(env):
    """Normalized action box by default; scaling and raw mode consistent."""
    from ags_env import AGSEnv, ACTION_BOUND
    assert env.norm_action, "default env should use normalized actions"
    assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0), \
        f"Expected normalized [-1, 1] box, got [{env.action_space.low[0]}, {env.action_space.high[0]}]"
    assert env.action_bound == ACTION_BOUND
    # out-of-range normalized action clips to +-ACTION_BOUND amperes inside step
    env.step(np.full(env.action_size, 2.0))
    assert np.all(env._dhc == ACTION_BOUND) and np.all(env._dvc == ACTION_BOUND), \
        "step() did not clip+scale normalized actions to ACTION_BOUND"
    env.reset()
    # raw mode: box mirrors ags_env.ACTION_BOUND in amperes
    raw = AGSEnv(norm_action=False)
    assert np.all(raw.action_space.low == -ACTION_BOUND) and \
           np.all(raw.action_space.high == ACTION_BOUND), \
        f"raw mode should expose +-{ACTION_BOUND}, got [{raw.action_space.low[0]}, {raw.action_space.high[0]}]"
    print(f"✓ Action box: [-1, 1] x {ACTION_BOUND} A (raw mode: +-{ACTION_BOUND} A)")


def test_with_gym_spaces():
    """Test that spaces are full float boxes compatible with gymnasium sampling."""
    from ags_env import AGSEnv

    env = AGSEnv()
    sample = env.action_space.sample()
    assert env.action_space.contains(sample), "action_space.sample() outside its own box"
    print("✓ gymnasium spaces consistent")


def test_lattice_seed():
    """Same lattice_seed -> identical frozen lattice; different seed -> different."""
    from ags_env import AGSEnv
    e1 = AGSEnv(lattice_seed=999)
    e2 = AGSEnv(lattice_seed=999)
    e3 = AGSEnv(lattice_seed=12345)
    r1 = e1.reset()[1]["reward"]
    r2 = e2.reset()[1]["reward"]
    r3 = e3.reset()[1]["reward"]
    assert r1 == r2, f"same lattice_seed gave different rewards: {r1} vs {r2}"
    assert e1.s_full == e2.s_full and e1.rms_ref == e2.rms_ref, "same seed changed reference constants"
    assert r1 != r3, f"different lattice_seed gave identical lattice: {r1} == {r3}"
    print(f"✓ Lattice seeding reproducible (seed 999 -> {r1:.2f}, seed 12345 -> {r3:.2f})")




def main():
    """Run all tests."""
    print("=" * 60)
    print("AGS Gymnasium Environment Test Suite")
    print("=" * 60)
    print()

    print("1. Testing environment construction...")
    env = test_env_construction()
    print()

    print("2. Testing spaces...")
    test_spaces(env)
    print()

    print("3. Testing BPM and control names...")
    test_bpm_and_control_names(env)
    print()

    print("4. Testing action bounds...")
    test_action_bounds(env)
    print()

    print("5. Testing reset...")
    test_reset(env)
    print()

    print("6. Testing step (zero action)...")
    test_step(env)
    print()

    print("7. Testing step (random action)...")
    test_random_action(env)
    print()

    print("8. Testing lattice seeding reproducibility...")
    test_lattice_seed()
    print()

    print("9. Testing gymnasium space consistency...")
    test_with_gym_spaces()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

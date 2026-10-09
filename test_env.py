"""
Test script for the AGS Gymnasium environment.

Run with: python test_env.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soct_integration import configure_julia

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
    """Test that action bounds are correct."""
    low = env.action_space.low
    high = env.action_space.high
    # bounds are defined by AGS_GymEnv.jl I_dhc/dvc_bounds (currently +/-25 A)
    assert np.all(low == -25.0), f"Expected low bound -25.0, got {low[0]}"
    assert np.all(high == 25.0), f"Expected high bound 25.0, got {high[0]}"
    print(f"✓ Action bounds: [{low[0]}, {high[0]}]")


def test_with_gym_spaces():
    """Test that spaces are full float boxes compatible with gymnasium sampling."""
    from ags_env import AGSEnv

    env = AGSEnv()
    sample = env.action_space.sample()
    assert env.action_space.contains(sample), "action_space.sample() outside its own box"
    print("✓ gymnasium spaces consistent")


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

    print("8. Testing gymnasium space consistency...")
    test_with_gym_spaces()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

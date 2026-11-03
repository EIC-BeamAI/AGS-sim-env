import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)

from soct_integration import configure_julia, register_env

configure_julia()  # must run before importing juliacall

import juliacall  # before TF, per ags_env.py note
import numpy as np
import jlab_opt_control.envs as eg

register_env(eg)
import jlab_opt_control.agents

# NOTE: default RUN is a PRE-REFACTOR (quad-error model, v0 id) run — pass a
# run directory produced by the current env as argv[1] for meaningful numbers.
# v0-era checkpoints are raw-ampere actors: NOT transferable to the shipped
# normalized-action space (evaluate them with norm_action=False semantics).
RUN = (sys.argv[1] if len(sys.argv) > 1 else
       "results/ags_td3_updated_env/"
       "index000_env_AGS-Injection-v0_agent_KerasTD3-v0_hash_a4105f9_time_20261008-174714")

env = eg.make('AGS-Injection-v1')
agent = jlab_opt_control.agents.make('KerasTD3-v0', env=env, logdir='/tmp/verify_logs')

LOADMAP = {
    'actor_model': 'actor_model', 'target_actor': 'target_actor',
    'critic_model1': 'critic_model1', 'critic_model2': 'critic_model2',
    'target_critic1': 'target_critic1', 'target_critic2': 'target_critic2',
}

for ckpt in ('epoch_09950_000', 'epoch_10000'):
    d = f"{RUN}/models/{ckpt}"
    import glob, os
    for f in sorted(glob.glob(f"{d}/*.weights.h5")):
        base = os.path.basename(f)
        attr = next(a for k, a in LOADMAP.items() if base.startswith(k + "_"))
        getattr(agent, attr).load_weights(f)
    print(f"[loaded {ckpt}]")
    state, info = env.reset()
    r0 = info['reward']
    a, _ = agent.action(state, train=False)
    a = np.asarray(a, dtype=np.float64).flatten()
    s2, r1, done, trunc, _ = env.step(a)
    S = s2[0:3 * 73:3]
    print(f"{ckpt}: zero-action reward {r0:.2f} -> policy reward {r1:.2f}")
    print(f"  S: min {S.min():.3f} max {S.max():.3f} (full survival = 10.014)")
    print(f"  |action|: max {np.abs(a).max():.2f}, at box edge {int((np.abs(a) > env.action_space.high[0] - 1e-9).sum())}/96")

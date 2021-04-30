import argparse
import os
import time

import gym
import numpy as np
import rc_gym
import torch

from agents.ddpg import DDPGActor
from agents.sac import GaussianPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-c_atk", "--checkpoint_atk", required=True,
                        help="checkpoint to load attacker")
    parser.add_argument("-c_gk", "--checkpoint_gk", required=True,
                        help="checkpoint to load goalkeeper")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    checkpoint_atk = torch.load(args.checkpoint_atk)
    checkpoint_gk = torch.load(args.checkpoint_gk)

    env = gym.make(checkpoint_atk['ENV_NAME'])

    if checkpoint_atk['AGENT'] == 'ddpg_async':
        pi_atk = DDPGActor(checkpoint_atk['N_OBS'], checkpoint_atk['N_ACTS']).to(device)
        pi_gk = DDPGActor(checkpoint_gk['N_OBS'], checkpoint_gk['N_ACTS']).to(device)
    elif checkpoint['AGENT'] == 'sac_async':
        pi_atk = GaussianPolicy(checkpoint_atk['N_OBS'], checkpoint_atk['N_ACTS'],
                            checkpoint_atk['LOG_SIG_MIN'],
                            checkpoint_atk['LOG_SIG_MAX'], checkpoint['EPSILON']).to(device)
        pi_gk = GaussianPolicy(checkpoint_gk['N_OBS'], checkpoint_gk['N_ACTS'],
                            checkpoint_gk['LOG_SIG_MIN'],
                            checkpoint_gk['LOG_SIG_MAX'], checkpoint_gk['EPSILON']).to(device)
    else:
        raise AssertionError

    pi_atk.load_state_dict(checkpoint_atk['pi_state_dict'])
    pi_gk.load_state_dict(checkpoint_gk['pi_state_dict'])
    pi_atk.eval()
    pi_gk.eval()

    while True:
        done = False
        s = env.reset()
        info = {}
        ep_steps = 0
        ep_rw_atk = 0
        ep_rw_gk = 0
        st_time = time.perf_counter()
        for i in range(checkpoint_atk['MAX_EPISODE_STEPS']):
            # Step the environment
            s_v_atk = torch.Tensor(s['observation_atk']).to(device)
            s_v_gk = torch.Tensor(s['observation_gk']).to(device)
            a_atk = pi_atk.get_action(s_v_atk)
            a_gk = pi_gk.get_action(s_v_gk)
            s_next, r, done, info = env.step({'action_atk': a_atk,'action_gk': a_gk})
            ep_steps += 1
            ep_rw_atk += r['reward_atk']
            ep_rw_gk += r['reward_gk']
            env.render()
            if done:
                break

            # Set state for next step
            s = s_next

        info['fps'] = ep_steps / (time.perf_counter() - st_time)
        info['ep_steps'] = ep_steps
        info['ep_rw_atk'] = ep_rw_atk
        info['ep_rw_gk'] = ep_rw_gk
        print(info)

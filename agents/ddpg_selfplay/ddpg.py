import torch
import collections
import time
import gym
import copy
import numpy as np
from agents.utils import NStepTracer, OrnsteinUhlenbeckNoise, generate_gif, HyperParameters, ExperienceFirstLast
import os
from dataclasses import dataclass


@dataclass
class DDPGHP(HyperParameters):
    AGENT: str = "ddpg_async"
    NOISE_SIGMA_INITIAL: float = None  # Initial action noise sigma
    NOISE_THETA: float = None
    NOISE_SIGMA_DECAY: float = None  # Action noise sigma decay
    NOISE_SIGMA_MIN: float = None
    NOISE_SIGMA_GRAD_STEPS: float = None  # Decay action noise every _ grad steps


def data_func(
    pi,
    device,
    queue_m,
    finish_event_m,
    sigma_m,
    gif_req_m,
    hp
):
    env = gym.make(hp.ENV_NAME)
    tracer_atk = NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)
    tracer_gk = NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low,
        max_value=env.action_space.high
    )

    with torch.no_grad():
        while not finish_event_m.is_set():
            # Check for generate gif request
            gif_idx = -1
            with gif_req_m.get_lock():
                if gif_req_m.value != -1:
                    gif_idx = gif_req_m.value
                    gif_req_m.value = -1
            if gif_idx != -1:
                path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}_atk.gif")
                generate_gif(env=env, filepath=path,
                             pi=copy.deepcopy(pi['pi_atk']), hp=hp)
                # verificar se faz diferença
                path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}_gk.gif")
                generate_gif(env=env, filepath=path,
                             pi=copy.deepcopy(pi['pi_gk']), hp=hp)

            done = False
            s = env.reset()
            noise.reset()
            if hp.MULTI_AGENT:
                [tracer[i].reset() for i in range(hp.N_AGENTS)]
            else:
                tracer.reset()
            noise.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            ep_rw_atk = 0
            ep_rw_gk = 0
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                s_v_atk = torch.Tensor(s).to(device)
                a_v_atk = pi(s_v_atk)
                a_atk = a_v_atk.cpu().numpy()
                # a_atk = noise(a_atk)

                s_v_gk = torch.Tensor(s).to(device)
                a_v_gk = pi(s_v_gk)
                a_gk = a_v_gk.cpu().numpy()
                # a_gk = noise(a_gk)

                s_next, r, done, info = env.step({'action_atk': a_atk, 'action_gk': a_gk})
                ep_steps += 1
                ep_rw_atk += r['reward_atk']
                ep_rw_gk += r['reward_gk']

                # Trace NStep rewards and add to mp queue
                exp_atk = ExperienceFirstLast(s['observation_atk'], a_atk, 
                                                r['reward_atk'], s_next['observation_atk'])
                
                exp_gk = ExperienceFirstLast(s['observation_gk'], a_atk, 
                                                r['reward_gk'], s_next['observation_gk'])
                
                queue_m.put({'exp_atk': exp_atk, 'exp_gk': exp_gk})
                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['noise'] = noise.sigma
            info['ep_steps'] = ep_steps
            info['ep_rw_atk'] = ep_rw_atk
            info['ep_rw_gk'] = ep_rw_gk
            queue_m.put(info)

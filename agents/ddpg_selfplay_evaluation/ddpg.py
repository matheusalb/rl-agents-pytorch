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
    tracer = NStepTracer(n=hp.REWARD_STEPS, gamma=hp.GAMMA)
    noise = OrnsteinUhlenbeckNoise(
        sigma=sigma_m.value,
        theta=hp.NOISE_THETA,
        min_value=env.action_space.low,
        max_value=env.action_space.high
    )

    pi_eval = pi['pi_eval']
    with torch.no_grad():
        while not finish_event_m.is_set():
            # Check for generate gif request
            gif_idx = -1
            with gif_req_m.get_lock():
                if gif_req_m.value != -1:
                    gif_idx = gif_req_m.value
                    gif_req_m.value = -1
            if gif_idx != -1:
                pass
                # path = os.path.join(hp.GIF_PATH, f"{gif_idx:09d}.gif")
                # generate_gif(env=env, filepath=path,
                #              pi=copy.deepcopy(pi), hp=hp)

            done = False
            s = env.reset()
            noise.reset()
            tracer.reset()
            noise.sigma = sigma_m.value
            info = {}
            ep_steps = 0
            ep_rw_eval = 0
            ep_rw_player = 0

            position_player = np.random.randint(0, len(pi['pi_player']))
            pi_player = pi['pi_player'][position_player]
            st_time = time.perf_counter()
            for i in range(hp.MAX_EPISODE_STEPS):
                # Step the environment
                
                # TODO
                # No momento supondo que o eval é o gk e o player é o atk, pensar melhor depois
                
                s_v_player= torch.Tensor(s['observation_atk']).to(device)
                a_player = pi_player(s_v_player)

                s_v_eval = torch.Tensor(s['observation_gk']).to(device)
                a_v_eval = pi_eval(s_v_eval)
                a_eval = a_v_eval.cpu().numpy()
                a_eval = noise(a_eval)

                s_next, r, done, info = env.step({'action_atk': a_player, 'action_gk': a_eval})
                ep_steps += 1
                ep_rw_player += r['reward_atk']
                ep_rw_eval += r['reward_gk']

                # Trace NStep rewards and add to mp queue
                tracer.add(s['observation_gk'], a_eval, r['reward_gk'], done)
                while tracer:
                    queue_m.put(tracer.pop())

                if done:
                    break

                # Set state for next step
                s = s_next

            info['fps'] = ep_steps / (time.perf_counter() - st_time)
            info['noise'] = noise.sigma
            info['ep_steps'] = ep_steps
            info['ep_rw_eval'] = ep_rw_eval
            queue_m.put(info)

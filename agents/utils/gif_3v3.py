import torch
import numpy as np
import PIL
import os
import gym
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction


def generate_gif_3v3(
    env, 
    filepath, 
    pi, 
    hp, 
    max_episode_steps=1200, 
    resize_to=None, 
    duration=25
):
    """
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment
    filepath : str
    pi : nn.Module
    max_episode_steps : int
    resize_to : tuple of ints, optional
    duration : float, optional
    """
    
    # collect frames
    frames = []
    s = env.reset()
    for t in range(max_episode_steps):
        actions = []
        for p, obs in zip(pi, s):
            if p is None:
                action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
                a = OrnsteinUhlenbeckAction(action_space, 0.025).sample()
            else:
                s_v = torch.Tensor(obs).to(hp.DEVICE)
                a = p.get_action(s_v)
            actions.append(a)
        s_next, r, done, info = env.step(actions)
        # store frame
        frame = env.render(mode='rgb_array')
        frame = PIL.Image.fromarray(frame)
        frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render(mode='rgb_array')
    frame = PIL.Image.fromarray(frame)
    frame = frame.convert('P', palette=PIL.Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    frames[0].save(
        fp=filepath, 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True,
        duration=duration, 
        loop=0
    )

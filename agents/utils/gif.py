import torch
import numpy as np
import PIL
import os

def generate_gif(
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
        s_v = torch.Tensor(s).to(hp.DEVICE)
        a = pi.get_action(s_v)
        s_next, r, done, info = env.step(a)
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

def generate_gif_selfplay(
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
        s_v_atk = torch.Tensor(s['observation_atk']).to(hp.DEVICE)
        s_v_gk = torch.Tensor(s['observation_gk']).to(hp.DEVICE)
        a_atk = pi['pi_atk'].get_action(s_v_atk)
        a_gk = pi['pi_gk'].get_action(s_v_gk)
        s_next, r, done, info = env.step({'action_atk': a_atk, 'action_gk': a_gk})
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
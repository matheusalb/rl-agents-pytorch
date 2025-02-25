import argparse
import copy
import dataclasses
import datetime
import os
import time

import gym
import numpy as np
import rc_gym
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

import wandb
from agents.ddpg_selfplay import (DDPGHP, DDPGActor, DDPGCritic, TargetActor,
                         TargetCritic, data_func)
from agents.utils import ReplayBuffer, save_checkpoint, unpack_batch, ExperienceFirstLast

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-e", "--env", required=True,
                        help="Name of the gym environment")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # Input Experiment Hyperparameters
    hp = DDPGHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=3,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=1,
        NOISE_SIGMA_INITIAL=0.8,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_MIN=0.15,
        NOISE_SIGMA_GRAD_STEPS=3000,
        REPLAY_SIZE=1000000,
        REPLAY_INITIAL=100000,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=100000,
        TOTAL_GRAD_STEPS=2000000,
        MULTI_AGENT=False,
        N_AGENTS=1
    )
    wandb.init(project='RoboCIn-RL', entity='matheusalb',
               name=hp.EXP_NAME, config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)

    pi_atk = DDPGActor(hp.N_OBS, hp.N_ACTS).to(device)
    Q_atk = DDPGCritic(hp.N_OBS, hp.N_ACTS).to(device)
    pi_gk = DDPGActor(hp.N_OBS, hp.N_ACTS).to(device)
    Q_gk = DDPGCritic(hp.N_OBS, hp.N_ACTS).to(device)

    # Playing
    pi_atk.share_memory()
    pi_gk.share_memory()
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    sigma_m = mp.Value('f', hp.NOISE_SIGMA_INITIAL)
    gif_req_m = mp.Value('i', -1)
    data_proc_list = []
    for _ in range(hp.N_ROLLOUT_PROCESSES):
        data_proc = mp.Process(
            target=data_func,
            args=(
                {'pi_atk': pi_atk, 'pi_gk': pi_gk},
                device,
                exp_queue,
                finish_event,
                sigma_m,
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    tgt_pi_atk = TargetActor(pi_atk)
    tgt_Q_atk = TargetCritic(Q_atk)
    tgt_pi_gk = TargetActor(pi_gk)
    tgt_Q_gk = TargetCritic(Q_gk)
    
    pi_opt_atk = optim.Adam(pi_atk.parameters(), lr=hp.LEARNING_RATE)
    Q_opt_atk = optim.Adam(Q_atk.parameters(), lr=hp.LEARNING_RATE)
    pi_opt_gk = optim.Adam(pi_gk.parameters(), lr=hp.LEARNING_RATE)
    Q_opt_gk = optim.Adam(Q_gk.parameters(), lr=hp.LEARNING_RATE)

    buffer_atk = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE
                          )
    
    buffer_gk = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE
                          )
    n_grads = 0
    n_samples = 0
    n_episodes = 0
    best_reward_atk = None
    best_reward_gk = None
    last_gif = None

    try:
        while n_grads < hp.TOTAL_GRAD_STEPS:
            metrics = {}
            ep_infos = list()
            st_time = time.perf_counter()
            # Collect EXP_GRAD_RATIO sample for each grad step
            new_samples = 0
            while new_samples < hp.EXP_GRAD_RATIO:
                exp = exp_queue.get()
                if exp is None:
                    raise Exception  # got None value in queue
                safe_exp = copy.deepcopy(exp)
                del(exp)

                # Dict is returned with end of episode info
                if not 'exp_atk' in safe_exp:
                    logs = {"ep_info/"+key: value for key,
                            value in safe_exp.items() if 'truncated' not in key}
                    ep_infos.append(logs)
                    n_episodes += 1
                else:
                    buffer_atk.add(
                    obs=safe_exp['exp_atk'].state,
                    next_obs=safe_exp['exp_atk'].last_state if safe_exp['exp_atk'].last_state is not None 
                            else safe_exp['exp_atk'].state,
                    action=safe_exp['exp_atk'].action,
                    reward=safe_exp['exp_atk'].reward,
                    done=False if safe_exp['exp_atk'].last_state is not None else True
                    )
                    
                    buffer_gk.add(
                    obs=safe_exp['exp_gk'].state,
                    next_obs=safe_exp['exp_gk'].last_state if safe_exp['exp_gk'].last_state is not None 
                            else safe_exp['exp_gk'].state,
                    action=safe_exp['exp_gk'].action,
                    reward=safe_exp['exp_gk'].reward,
                    done=False if safe_exp['exp_gk'].last_state is not None else True
                    )
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()


            # Only start training after buffer is larger than initial value
            if buffer_atk.size() < hp.REPLAY_INITIAL or buffer_gk.size() < hp.REPLAY_INITIAL:
                continue

            # Sample a batch and load it as a tensor on device
            batch_atk = buffer_atk.sample(hp.BATCH_SIZE)
            S_v_atk = batch_atk.observations
            A_v_atk = batch_atk.actions
            r_v_atk = batch_atk.rewards
            dones_atk = batch_atk.dones
            S_next_v_atk = batch_atk.next_observations
            
            batch_gk = buffer_gk.sample(hp.BATCH_SIZE)
            S_v_gk = batch_gk.observations
            A_v_gk = batch_gk.actions
            r_v_gk = batch_gk.rewards
            dones_gk = batch_gk.dones
            S_next_v_gk = batch_gk.next_observations

            # train critic
            Q_opt_atk.zero_grad()
            Q_v_atk = Q_atk(S_v_atk, A_v_atk)  # expected Q for S,A
            A_next_v_atk = tgt_pi_atk(S_next_v_atk)  # Get an Bootstrap Action for S_next
            Q_next_v_atk = tgt_Q_atk(S_next_v_atk, A_next_v_atk)  # Bootstrap Q_next
            Q_next_v_atk[dones_atk == 1.] = 0.0  # No bootstrap if transition is terminal
            # Calculate_atk a reference Q value using the bootstrap Q
            Q_ref_v_atk = r_v_atk + Q_next_v_atk * (hp.GAMMA**hp.REWARD_STEPS)
            Q_loss_v_atk = F.mse_loss(Q_v_atk, Q_ref_v_atk.detach())
            Q_loss_v_atk.backward()
            Q_opt_atk.step()
            metrics["train/loss_Q_atk"] = Q_loss_v_atk.cpu().detach().numpy()
    
            Q_opt_gk.zero_grad()
            Q_v_gk = Q_gk(S_v_gk, A_v_gk)  # expected Q for S,A
            A_next_v_gk = tgt_pi_gk(S_next_v_gk)  # Get an Bootstrap Action for S_next
            Q_next_v_gk = tgt_Q_gk(S_next_v_gk, A_next_v_gk)  # Bootstrap Q_next
            Q_next_v_gk[dones_gk == 1.] = 0.0  # No bootstrap if transition is terminal
            # Calculate_gk a reference Q value using the bootstrap Q
            Q_ref_v_gk = r_v_gk + Q_next_v_gk * (hp.GAMMA**hp.REWARD_STEPS)
            Q_loss_v_gk = F.mse_loss(Q_v_gk, Q_ref_v_gk.detach())
            Q_loss_v_gk.backward()
            Q_opt_gk.step()
            metrics["train/loss_Q_gk"] = Q_loss_v_gk.cpu().detach().numpy()

            # train actor - Maximize Q value received over every S
            pi_opt_atk.zero_grad()
            A_cur_v_atk = pi_atk(S_v_atk)
            pi_loss_v_atk = -Q_atk(S_v_atk, A_cur_v_atk)
            pi_loss_v_atk = pi_loss_v_atk.mean()
            pi_loss_v_atk.backward()
            pi_opt_atk.step()
            metrics["train/loss_pi_atk"] = pi_loss_v_atk.cpu().detach().numpy()
            
            pi_opt_gk.zero_grad()
            A_cur_v_gk = pi_gk(S_v_gk)
            pi_loss_v_gk = -Q_gk(S_v_gk, A_cur_v_gk)
            pi_loss_v_gk = pi_loss_v_gk.mean()
            pi_loss_v_gk.backward()
            pi_opt_gk.step()
            metrics["train/loss_pi_gk"] = pi_loss_v_gk.cpu().detach().numpy()

            # Sync target networks
            tgt_pi_atk.sync(alpha=1 - 1e-3)
            tgt_Q_atk.sync(alpha=1 - 1e-3)
            
            tgt_pi_gk.sync(alpha=1 - 1e-3)
            tgt_Q_gk.sync(alpha=1 - 1e-3)

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics['counters/buffer_len_atk'] = buffer_atk.size()
            metrics['counters/buffer_len_gk'] = buffer_gk.size()

            if ep_infos:
                for key in ep_infos[0].keys():
                    metrics[key] = np.mean([info[key] for info in ep_infos])

            # Log metrics
            wandb.log(metrics)

            if hp.NOISE_SIGMA_DECAY and sigma_m.value > hp.NOISE_SIGMA_MIN \
                and n_grads % hp.NOISE_SIGMA_GRAD_STEPS == 0:
                # This syntax is needed to be process-safe
                # The noise sigma value is accessed by the playing processes
                with sigma_m.get_lock():
                    sigma_m.value *= hp.NOISE_SIGMA_DECAY

            if hp.SAVE_FREQUENCY and n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    hp=hp,
                    metrics={
                        'noise_sigma': sigma_m.value,
                        'n_samples': n_samples,
                        'n_episodes': n_episodes,   
                        'n_grads': n_grads,
                    },
                    pi=pi_atk,
                    Q=Q_atk,
                    pi_opt=pi_opt_atk,
                    Q_opt=Q_opt_atk,
                    label="atk"
                )

                save_checkpoint(
                    hp=hp,
                    metrics={
                        'noise_sigma': sigma_m.value,
                        'n_samples': n_samples,
                        'n_episodes': n_episodes,   
                        'n_grads': n_grads,
                    },
                    pi=pi_gk,
                    Q=Q_gk,
                    pi_opt=pi_opt_gk,
                    Q_opt=Q_opt_gk,
                    label="gk"
                )

            if hp.GIF_FREQUENCY and n_grads % hp.GIF_FREQUENCY == 0:
                gif_req_m.value = n_grads

    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    finally:
        if exp_queue:
            while exp_queue.qsize() > 0:
                exp_queue.get()

        print('queue is empty')

        print("Waiting for threads to finish...")
        for p in data_proc_list:
            p.terminate()
            p.join()

        del(exp_queue)
        del(pi_atk)
        del(pi_gk)

        finish_event.set()

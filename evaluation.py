import argparse
import copy
import dataclasses
import datetime
import os
import time

import gym
import numpy as np
import rc_gym
import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim



import wandb
from agents.ddpg_selfplay_evaluation import (DDPGHP, DDPGActor, DDPGCritic, TargetActor,
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
    parser.add_argument("-c_eval", "--checkpoint_eval", required=True,
                        help="checkpoint to load the policy to be evaluated")
    parser.add_argument("-p_players", "--path_players", default="./atk_policies",
                        required=False,help="path of policies to be used in the evaluation")
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # Input Experiment Hyperparameters
    hp = DDPGHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=2,
        LEARNING_RATE=0.0001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=3,
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

    # Lendo as politicas treinadas dos players
    l_pls = os.listdir(args.path_players)
    if len(l_pls) == 0:
        raise Exception("No player policy found in ./atk_policies")
    
    checkpoint_player = []
    for p in l_pls:
        checkpoint_player.append(torch.load(args.path_players +'/'+ p))

    pi_players = []
    for cpplayer in checkpoint_player:
        pi_players.append(DDPGActor(cpplayer['N_OBS'], cpplayer['N_ACTS']).to(device))
        pi_players[-1].load_state_dict(cpplayer['pi_state_dict'])
        pi_players[-1].eval()

    # Lendo a politica a ser avaliada treinado
    checkpoint_eval = torch.load(args.checkpoint_eval)
    pi_eval = DDPGActor(checkpoint_eval['N_OBS'], checkpoint_eval['N_ACTS']).to(device)
    Q_eval = DDPGCritic(checkpoint_eval['N_OBS'], checkpoint_eval['N_ACTS']).to(device)

    pi_eval.load_state_dict(checkpoint_eval['pi_state_dict'])
    Q_eval.load_state_dict(checkpoint_eval['Q_state_dict'])

    # Playing
    pi_eval.share_memory()
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    sigma_m = mp.Value('f', hp.NOISE_SIGMA_INITIAL)
    gif_req_m = mp.Value('i', -1)
    data_proc_list = []
    for _ in range(hp.N_ROLLOUT_PROCESSES):
        data_proc = mp.Process(
            target=data_func,
            args=(
                {'pi_eval': pi_eval, 'pi_player': pi_players},
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
    tgt_pi_eval = TargetActor(pi_eval)
    tgt_Q_eval = TargetCritic(Q_eval)
    
    pi_opt_eval = optim.Adam(pi_eval.parameters(), lr=hp.LEARNING_RATE)
    Q_opt_eval = optim.Adam(Q_eval.parameters(), lr=hp.LEARNING_RATE)
    
    buffer = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE
                          )
    n_grads = 0
    n_samples = 0
    n_episodes = 0
    best_reward_eval = None
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
                if isinstance(safe_exp, dict):
                    logs = {"ep_info/"+key: value for key,
                            value in safe_exp.items() if 'truncated' not in key}
                    ep_infos.append(logs)
                    n_episodes += 1
                else:
                    buffer.add(
                        obs=safe_exp.state,
                        next_obs=safe_exp.last_state if safe_exp.last_state is not None 
                                else safe_exp.state,
                        action=safe_exp.action,
                        reward=safe_exp.reward,
                        done=False if safe_exp.last_state is not None else True
                    )
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()


            # Only start training after buffer is larger than initial value
            if buffer.size() < hp.REPLAY_INITIAL:
                continue

            # Sample a batch and load it as a tensor on device
            batch = buffer.sample(hp.BATCH_SIZE)
            S_v_eval = batch.observations
            A_v_eval = batch.actions
            r_v_eval = batch.rewards
            dones = batch.dones
            S_next_v_eval = batch.next_observations
            
            # train critic
            Q_opt_eval.zero_grad()
            Q_v_eval = Q_eval(S_v_eval, A_v_eval)  # expected Q for S,A
            A_next_v_eval = tgt_pi_eval(S_next_v_eval)  # Get an Bootstrap Action for S_next
            Q_next_v_eval = tgt_Q_eval(S_next_v_eval, A_next_v_eval)  # Bootstrap Q_next
            Q_next_v_eval[dones == 1.] = 0.0  # No bootstrap if transition is terminal
            # Calculate a reference Q value using the bootstrap Q
            Q_ref_v_eval = r_v_eval + Q_next_v_eval * (hp.GAMMA**hp.REWARD_STEPS)
            Q_loss_v_eval = F.mse_loss(Q_v_eval, Q_ref_v_eval.detach())
            Q_loss_v_eval.backward()
            Q_opt_eval.step()
            metrics["train/loss_Q_eval"] = Q_loss_v_eval.cpu().detach().numpy()

            # train actor - Maximize Q value received over every S
            pi_opt_eval.zero_grad()
            A_cur_v_eval = pi_eval(S_v_eval)
            pi_loss_v_eval = -Q_eval(S_v_eval, A_cur_v_eval)
            pi_loss_v_eval = pi_loss_v_eval.mean()
            pi_loss_v_eval.backward()
            pi_opt_eval.step()
            metrics["train/loss_pi_eval"] = pi_loss_v_eval.cpu().detach().numpy()
            
            # Sync target networks
            tgt_pi_eval.sync(alpha=1 - 1e-3)
            tgt_Q_eval.sync(alpha=1 - 1e-3)
            
            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics['counters/buffer_len'] = buffer.size()

            if ep_infos:
                for key in ep_infos[0].keys():
                    # print(ep_infos[0])
                    metrics[key] = np.mean([info[key] for info in ep_infos if key != 'ep_info/energy_atk'])

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
                    pi=pi_eval,
                    Q=Q_eval,
                    pi_opt=pi_opt_eval,
                    Q_opt=Q_opt_eval,
                    label="eval"
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
        del(pi_eval)

        finish_event.set()

from __future__ import print_function
import numpy as np
import torch
import os
import utils
import TD3
import json
import time
from tensorboardX import SummaryWriter
from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import checkpoint as cp
from config import *


def train(args):

    # Set up directories ===========================================================
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
    exp_name = "EXP_%04d" % (args.expID)
    exp_path = os.path.join(DATA_DIR, exp_name)
    rb_path = os.path.join(BUFFER_DIR, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(rb_path, exist_ok=True)
    # Save arguments.
    with open(os.path.join(exp_path, 'args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    # Retrieve MuJoCo XML files for training ========================================
    envs_train_names = []
    args.graphs = dict()
    # Existing envs.
    if not args.custom_xml:
        for morphology in args.morphologies:
            envs_train_names += [name[:-4] for name in os.listdir(XML_DIR) if '.xml' in name and morphology in name]
        for name in envs_train_names:
            args.graphs[name] = utils.getGraphStructure(os.path.join(XML_DIR, '{}.xml'.format(name)))
    # Custom envs.
    else:
        if os.path.isfile(args.custom_xml):
            assert '.xml' in os.path.basename(args.custom_xml), "No xml file found."
            name = os.path.basename(args.custom_xml)
            envs_train_names.append(name[:-4])
            args.graphs[name[:-4]] = utils.getGraphStructure(args.custom_xml)
        elif os.path.isdir(args.custom_xml):
            for name in os.listdir(args.custom_xml):
                if '.xml' in name:
                    envs_train_names.append(name[:-4])
                    args.graphs[name[:-4]] = utils.getGraphStructure(os.path.join(args.custom_xml, name))
    envs_train_names.sort()
    num_envs_train = len(envs_train_names)

    print("#" * 50 + '\ntraining envs: {}\n'.format(envs_train_names) + "#" * 50)

    # Set up training env and policy ================================================
    args.limb_obs_size, args.max_action = utils.registerEnvs(envs_train_names, args.max_episode_steps, args.custom_xml)
    max_num_limbs = max([len(args.graphs[env_name]) for env_name in envs_train_names])
    # Create vectorized training env.
    obs_max_len = max([len(args.graphs[env_name]) for env_name in envs_train_names]) * args.limb_obs_size
    envs_train = [utils.makeEnvWrapper(name, obs_max_len, args.seed) for name in envs_train_names]
    # Vectorized env.
    envs_train = SubprocVecEnv(envs_train)
    # Set random seeds.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Determine the maximum number of children in all the training envs.
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(envs_train_names, args.graphs)
    # setup agent policy
    policy = TD3.TD3(args)

    # Create new training instance or load previous checkpoint ========================
    if cp.has_checkpoint(exp_path):
        print("*** loading checkpoint from {} ***".format(exp_path))
        total_timesteps, life_num, num_samples, loaded_path = cp.load_checkpoint_without_rb(exp_path, policy, args)
        print("*** checkpoint loaded from {} ***".format(loaded_path))
    else:
        print("*** training from scratch ***")
        # Init training vars.
        total_timesteps = 0
        num_samples = 0
        life_num = 0

    # Different replay buffer for each env; avoid using too much memory if there are too many envs.
    replay_buffer = dict()
    if num_envs_train > args.rb_max // 1e6:
        for name in envs_train_names:
            replay_buffer[name] = utils.ReplayBufferWithMemory(max_size=args.rb_max // num_envs_train)
    else:
        for name in envs_train_names:
            replay_buffer[name] = utils.ReplayBufferWithMemory()

    # Initialize training variables ================================================
    writer = SummaryWriter("%s/%s/" % (DATA_DIR, exp_name))
    life_num_since_saving = 0
    life_done = False
    life_time_step = 0
    life_reward_list = np.zeros(num_envs_train)  # Convert list to a 1D array
    episode_num = np.zeros(num_envs_train)  # Convert list to a 1D integer array
    episode_timesteps_list = np.zeros(num_envs_train)  # Convert list to a 1D integer array
    mean_episode_len = np.zeros(num_envs_train)  # Convert list to a 1D array
    old_action_list = np.zeros((num_envs_train, max_num_limbs))  # Convert list to a 2D array
    old_h_list = np.zeros((num_envs_train, 128 * max_num_limbs))  # Convert list to a 2D array
    old_reward_list = np.zeros((num_envs_train, max_num_limbs))  # Convert list to a 2D array
    old_obs_list = envs_train.reset()

    # Start training ===========================================================
    while life_num < args.max_life_num:
        # Sampling ==========================================
        action_list = []
        h_list = []
        sample_start_time = time.time()
        while not life_done:
            # Forward.
            life_time_step += 1
            if life_time_step == args.life_time_steps:
                life_done = True
            for i in range(num_envs_train):
                # Dynamically change the graph structure of the modular policy.
                policy.change_morphology(args.graphs[envs_train_names[i]])
                # remove 0 padding of obs before feeding into the policy (trick for vectorized env)
                obs = np.array(old_obs_list[i][:args.limb_obs_size * len(args.graphs[envs_train_names[i]])])               
                action_0 = np.array(old_action_list[i][:len(args.graphs[envs_train_names[i]])])
                r_0 = np.array(old_reward_list[i][:len(args.graphs[envs_train_names[i]])])
                h = np.array(old_h_list[i][:128*len(args.graphs[envs_train_names[i]])])
                policy_action, policy_h= policy.select_action(obs, np.array(action_0), np.array(r_0), np.array(h))
                if life_num == 0:
                    policy_action = (policy_action + np.random.normal(0, 0.3,
                        size=policy_action.size)).clip(envs_train.action_space.low[0],
                        envs_train.action_space.high[0])
                else:
                    if args.expl_noise != 0:
                        policy_action = (policy_action + np.random.normal(0, args.expl_noise,
                            size=policy_action.size)).clip(envs_train.action_space.low[0],
                            envs_train.action_space.high[0])
                # Add 0-padding to ensure that size is the same for all envs.
                policy_action = np.append(policy_action, np.array([0 for i in range(max_num_limbs - policy_action.size)]))
                policy_h = np.append(policy_h, np.array([0 for i in range(max_num_limbs - policy_h.size)]))
                action_list.append(policy_action)
                h_list.append(policy_h)

            # Perform action in enviroments.
            new_obs_list, reward_list, curr_done_list, _ = envs_train.step(action_list)
            reward_list = np.array(reward_list)
            reward_list = np.repeat(reward_list[:, np.newaxis], max_num_limbs, axis=1)
            # Record if each env has ever been 'done'.
            done_list = [curr_done_list[i] for i in range(num_envs_train)]

            # Saving into replaybuffer.
            for i in range(num_envs_train):
                writer.add_scalar('{}_instant_reward'.format(envs_train_names[i]), reward_list[i, 0], total_timesteps)
                life_reward_list[i] += reward_list[i, 0]
                # Add to replay buffer.
                if episode_timesteps_list[i] == args.max_episode_steps:
                    done_list[i] = True
                # Remove 0 padding before storing in the replay buffer (trick for vectorized env).
                num_limbs = len(args.graphs[envs_train_names[i]])
                obs = np.array(old_obs_list[i][:args.limb_obs_size * num_limbs])
                new_obs = np.array(new_obs_list[i][:args.limb_obs_size * num_limbs])
                action = np.array(action_list[i][:num_limbs])
                old_action = np.array(old_action_list[i][:num_limbs])
                old_reward = np.array(old_reward_list[i][:num_limbs])
                reward= np.array(reward_list[i][:num_limbs])
                h = np.array(h_list[i][:128*num_limbs])
                # Insert data into the replay buffer.
                data = (obs, new_obs, old_action, action, old_reward, reward, h, done_list[i])
                replay_buffer[envs_train_names[i]].add(data , life_done)
                episode_timesteps_list[i] += 1
                # Reset env if this env is done.
                if done_list[i]:
                    envs_train.remotes[i].send(("reset", None))
                    new_obs_list[i] = envs_train.remotes[i].recv()
                    episode_timesteps_list[i] = 0
                    action_list[i] = np.zeros(max_num_limbs)
                    reward_list[i] = np.zeros(max_num_limbs)
                    episode_num[i] += 1

                total_timesteps += 1
                num_samples += 1

            # Save old action, reward, h
            old_obs_list = new_obs_list
            old_action_list = action_list
            old_h_list = h_list
            old_reward_list = reward_list

        for i in range(num_envs_train):
            mean_episode_len[i] = (life_time_step - episode_timesteps_list[i]) / episode_num[i]
            print("mean_episode_len ", mean_episode_len[i])
            # Add to tensorboard display.
            writer.add_scalar('{}_episode_reward'.format(envs_train_names[i]), life_reward_list[i], life_num)
            writer.add_scalar('{}_episode_len'.format(envs_train_names[i]), mean_episode_len[i], life_num)

        life_num += 1
        life_num_since_saving += 1
        # Print to console.
        print("-" * 50 + "\nExpID: {}, TotalT: {}, Life_num: {}".format(
                args.expID,
                total_timesteps,
                life_num))
        for i in range(len(envs_train_names)):
            print("{} === EpisodeNum: {}, Reward: {:.2f}".format(envs_train_names[i],
                                                               episode_num[i],
                                                               life_reward_list[i]))

        # reset training variables
        old_obs_list = envs_train.reset()
        life_time_step = 0
        life_reward_list = np.zeros(num_envs_train) 
        episode_timesteps_list =  np.zeros(num_envs_train)
        episode_num = np.zeros(num_envs_train)
        mean_episode_len = np.zeros(num_envs_train)
        old_action_list = np.zeros((num_envs_train, max_num_limbs)) 
        old_h_list = np.zeros((num_envs_train, 128*max_num_limbs)) 
        old_reward_list = np.zeros((num_envs_train, max_num_limbs)) 

        life_done = False
        sample_end_time = time.time()
        sample_time = sample_end_time - sample_start_time
        print(f"sample time: {sample_time:.6f} sec")

        # Collect done, start training =================================================
        if life_num_since_saving == args.train_freq:
            # Log updates and train policy.
            start_train_time = time.time()
            policy.train(replay_buffer, args.life_time_steps, num_envs_train,
                        args.discount, args.tau, args.policy_noise, args.noise_clip,
                        args.policy_freq, graphs=args.graphs, envs_train_names=envs_train_names[:num_envs_train])
            end_train_time = time.time()
            train_time = end_train_time - start_train_time
            print(f"train time: {train_time:.6f} sec")

            for name in envs_train_names:
                replay_buffer[name].clear()
            life_num_since_saving = 0

            # Save model and replay buffers when training done.
            model_saved_path = cp.save_model_without_rb(exp_path, policy, total_timesteps,
                                             life_num, num_samples,
                                             envs_train_names, args)
            print("*** model saved to {} ***".format(model_saved_path))


if __name__ == "__main__":
    args = get_args()
    train(args)

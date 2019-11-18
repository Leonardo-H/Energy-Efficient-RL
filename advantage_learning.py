import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf
import tempfile
import time
from collections import deque

import baselines.common.tf_util as U

from baselines import logger
from baselines import EERL
from baselines.EERL.replay_buffer import ReplayBuffer
from baselines.common.misc_util import (
    set_global_seeds,
    SimpleMonitor
)

from baselines.common.schedules import PiecewiseSchedule
# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.EERL.env_init import wrap_dqn_bnn
from model import v_func_model, binary_model_1

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    
    # Basic parameters
    parser.add_argument("--en", type=int, default=4, help="the number of ensemble BNNs")
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=50, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=25000, help="number of iterations between every target network update")

    # BNN evaluation setting
    parser.add_argument("--eval", action='store_true', help="whether to evaluate model")
    parser.add_argument("--eval-freq", type=int, default=1e5, help="number of steps between every evaluation")
    parser.add_argument("--eval-times", type=int, default=30, help="how many episodes to evaluate in one evaluation time")

    parser.add_argument("--save-model", action='store_true', help="whether to save-model")
    parser.add_argument("--use-sign", action='store_true', help="whether to use sign")
    return parser.parse_args()

def make_env(game_name, noop_max):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn_bnn(monitored_env, noop_max)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env
    
if __name__ == '__main__':
    args = parse_args()

    env_name = args.env
    seed = args.seed

    end_fix = '_seed={}_{}'.format(seed, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    file_name='./adv_learning/{}/num={}'.format(env_name, args.en,)
    save_path = './adv_learning/{}/saver_num={}'.format(env_name, args.en)

    file_name += end_fix
    
    # Create and seed the env.
    if args.eval:
        for i in range(100):
            print('We will use evaluation methods')        
        # if evaluation, train env do not use no-op
        env, monitored_env = make_env(args.env, 1)
        # but eval env use random no-op before each episodes.
        eval_env, eval_monitored_env = make_env(args.env, 30)
    else:
        for i in range(100):
            print('We will not use evaluation methods')
        env, monitored_env = make_env(args.env, 1)

    if args.seed >= 0:
        set_global_seeds(args.seed)
        env.unwrapped.seed(args.seed)
        if args.eval:
            eval_env.unwrapped.seed(args.seed)

    done = False
    obs = env.reset()
    info = {'rewards': []}
    train_episodes = 0

    if args.eval:
        eval_done = False
        eval_obs = eval_env.reset()
        eval_info = {'rewards': []}
        eval_score = 0

    last_eval_steps = 0

    with U.make_session(4) as sess:
        act, train, update_target = EERL.adv_build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            v_func=v_func_model,
            adv_func=binary_model_1,
            learning_rate=args.lr,
            num_actions=env.action_space.n,
            en=args.en,

            gamma=0.99,
            grad_norm_clipping=10,
        )

        approximate_num_iters = args.num_steps / 4
        exploration = PiecewiseSchedule([
            (0, 1.0),
            (approximate_num_iters / 50, 0.1),
            (approximate_num_iters / 5, 0.01)
        ], outside_value=0.01)

        replay_buffer = ReplayBuffer(args.replay_buffer_size)
        
        U.initialize()
        
        update_target()
        num_iters = 0        

        if not os.path.exists('./adv_learning/' + env_name):
            os.makedirs('./adv_learning/' + env_name)

        if args.save_model:
            saver = tf.train.Saver()
            if not os.path.exists(os.path.join(save_path, 'Saver', 'Model')):
                os.makedirs(os.path.join(save_path, 'Saver', 'Model'))
            saver.save(sess, os.path.join(save_path, 'Saver', 'Model'))


        with open(file_name + '_record.txt', 'a') as f:
            print(sys.executable + " " + " ".join(sys.argv), file=f)
            print(vars(args), file=f)

        # Main trianing loop
        while True:
            num_iters += 1
            
            if args.save_model and (num_iters+1)%100000==0:
                saver.save(sess, os.path.join(save_path, 'Saver', 'Model'), write_meta_graph=False)

            # Take action and store transition in the replay buffer.
            action, a_values = act(np.array(obs)[None], 
                        update_eps=exploration.value(num_iters),
                        is_training=False,)

            #TODO
            if num_iters % 1000 == 0:
                print(a_values)

            new_obs, rew, done, info = env.step(action)
            assert rew == 1.0 or rew == 0.0 or rew == -1.0

            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            if done:
                obs = env.reset()

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                    
                # Sample a batch of transitions from replay buffer                
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size * args.en)

                # Minimize the error in Bellman's equation and compute TD-error
                total_td_errors = train(obses_t, actions, rewards, obses_tp1, dones)

            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if info["steps"] > args.num_steps:
                break

            #TODO
            if num_iters % 5000 == 0:
                record = {
                        'steps': info["steps"],
                        'mean_100': np.mean(info['rewards'][-101:-1]),
                        'mean_50': np.mean(info['rewards'][-51:-1]),
                    }
                if len(info['rewards']) > 2:
                    record['last_reward'] = info['rewards'][-2]
                if args.eval:
                    record['eval_reward'] = eval_score

                with open(file_name + '_record.txt', 'a') as f:                 
                    print(record, file=f)

            # show log information only if one episode is end
            if done and len(info['rewards']) > train_episodes:
                train_episodes += 1
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))        

                #TODO
                if len(info['rewards']) > 2:
                    logger.record_tabular("last reward", info["rewards"][-2])    
                logger.record_tabular("recent reward (50 epi mean)", np.mean(info["rewards"][-51:-1]))

                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-101:-1]))
                if args.eval:
                    logger.record_tabular("evaluation score", eval_score)

                logger.record_tabular("exploration", exploration.value(num_iters))

                logger.dump_tabular()
                logger.log()

            # evaluation
            if args.eval and info['steps'] > last_eval_steps + args.eval_freq:
                
                history_episodes_num = len(eval_info['rewards'])
                last_eval_steps = info['steps']
                
                start_episode = len(eval_info['rewards'])
                while 1:
                    if eval_done:
                        eval_obs = eval_env.reset()
                        if len(eval_info['rewards']) >= history_episodes_num + args.eval_times:
                            eval_score = np.mean(eval_info['rewards'][history_episodes_num:])
                            assert len(eval_info['rewards'][history_episodes_num:]) == args.eval_times
                            break
                    
                    eval_act = act(np.array(eval_obs)[None])[0]
                    eval_obs, r, eval_done, eval_info = eval_env.step(eval_act)
                print(eval_info['rewards'][history_episodes_num:])

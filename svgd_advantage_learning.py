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
# copy over LazyFrames
from baselines.EERL.env_init import wrap_dqn_bnn
from model import v_func_model, binary_model_1


def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Breakout", help="name of the game")
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=50, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    parser.add_argument("--target-update-freq", type=int, default=25000, help="number of iterations between every target network update")
    
    # basic parameters
    parser.add_argument("--en", type=int, default=4, help="how many BNN network we need")
    
    # BNN evaluation setting
    parser.add_argument("--eval", action='store_true', help="whether to evaluate model")
    parser.add_argument("--eval-freq", type=int, default=1e5, help="after which step evaluate the BNN once")
    parser.add_argument("--eval-times", type=int, default=30, help="how many times evaluate the network in one evaluation time")

    # save or load model parameters
    parser.add_argument("--save-model", action='store_true', help="whether to save the model")
    
    parser.add_argument("--tau", type=float, default=1, help="svgd temperature")
    parser.add_argument("--max-iters", type=int, default=int(5e6), help="the max iters when tau decay to 0")
    parser.add_argument("--h-tau", type=float, default=1, help="svgd h temperature")

    parser.add_argument("--need-probe", action='store_true', help="whether we need to probe")
    parser.add_argument("--decay", action='store_true', help="whether we need to probe")

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
    args.need_probe = True

    end_fix = '_seed={}_{}'.format(seed, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    file_name='./svgd_adv_learning/{}/svgd_en={}'.format(env_name, args.en)
    save_path = './svgd_adv_learning/{}/svgd_saver_en={}'.format(env_name, args.en)

    save_path += end_fix
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
        # Create training graph and replay buffer
        # stable target means use averaged ensemble BNN values as target advantage rewards
        
        act, train, update_target, debug = EERL.svgd_adv_build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            v_func=v_func_model,
            adv_func=binary_model_1,
            learning_rate=args.lr,
            num_actions=env.action_space.n,
            en=args.en,

            gamma=0.99,
            grad_norm_clipping=10,
        )
        predict_values = debug['predict_values']
        
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
        done_times = 0
                
        if not os.path.exists('./svgd_adv_learning/' + env_name):
            os.makedirs('./svgd_adv_learning/' + env_name)

        obs = env.reset()

        # records setting:
        record_set = {
            'eval-rewards': [],
            'TD-Error': [],
            'D-mid': [],
            'h': [],
        }
        
        for i in range(args.en):
            record_set['eval-rewards'].append(deque(maxlen=200))
            record_set['TD-Error'].append(deque(maxlen=200))

        max_eval_rewards = 0
        max_rewards = 0
        train_times = 0
        save_message = []        
        eval_info = {'rewards':[0], 'ale.lives':5, 'steps':0}
        last_death_steps = 0
        r_unwritten_list = []
        s_unwritten_list = []
        eval_start = 0

        if args.save_model:
            saver = tf.train.Saver()
            if not os.path.exists(os.path.join(save_path, 'Saver', 'Model')):
                os.makedirs(os.path.join(save_path, 'Saver', 'Model'))
            saver.save(sess, os.path.join(save_path, 'Saver', 'Model'))

        with open(file_name + '_record.txt', 'a') as f:
            print(sys.executable + " " + " ".join(sys.argv), file=f)
            print(vars(args), file=f)

        # Main trianing loop
        current_eval_times = 0
        D_mid = 0
        h = 0

        tau = args.tau
        h_tau = args.h_tau
        while True:
            num_iters += 1
            if args.save_model and (num_iters+1)%5000==0:
                saver.save(sess, os.path.join(save_path, 'Saver', 'Model'), write_meta_graph=False)
                
            # Take action and store transition in the replay buffer.
            act_func_output = act(np.array(obs)[None], 
                        update_eps=exploration.value(num_iters),
                        is_training=False,)            
            action, a_values, a_values_list = act_func_output[0], act_func_output[1], act_func_output[2:]

            new_obs, rew, done, info = env.step(action)

            if num_iters % 10000 == 0:
                predict_values_outputs = predict_values(
                    np.array([np.array(obs)] * args.en), 
                    np.array([action] * args.en), 
                    np.array([rew] * args.en), 
                    np.array([np.array(new_obs)] * args.en), 
                    np.array([float(done)] * args.en)
                )
                print('action choice is', action)
                print('reward get is', rew)
                print('v is ', predict_values_outputs[0])
                print('adv is ')
                for i in range(args.en):
                    print(predict_values_outputs[2 + 0 * args.en + i])
                print('q is')
                for i in range(args.en):
                    print(predict_values_outputs[2 + 2 * args.en + i])
                print('')
                print('target v is ', predict_values_outputs[1])
                print('target adv is')
                for i in range(args.en):
                    print(predict_values_outputs[2 + 1 * args.en + i])
                print('target q is')
                for i in range(args.en):
                    print(predict_values_outputs[2 + 3 * args.en + i])
            
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            if done:
                obs = env.reset()
                done_times += 1

            if (num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and
                    num_iters % args.learning_freq == 0):
                train_times += 1
                # Sample a bunch of transitions from replay buffer
                
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size * args.en)

                # Minimize the error in Bellman's equation and compute TD-error
                outputs = train(obses_t, actions, rewards, obses_tp1, dones, tau, h_tau)

                for i in range(args.en):
                    record_set['TD-Error'][i].append(outputs[i])
                            
                if args.need_probe and num_iters % (args.learning_freq * 5000) == 0:
                    probes = outputs[args.en : ]
                    D, D_mid, h, kernel = probes

                    print('D is \n', D)
                    print('Dmid is ', D_mid, 'h is ', h)
                    print('kernel is \n', kernel)
                    
            # Update target network.
            if num_iters % args.target_update_freq == 0:
                update_target()

            if info["steps"] > args.num_steps:
                break

            if num_iters % 5000 == 0:
                record = {
                        'steps': info["steps"],
                        'mean_100': np.mean(info['rewards'][-101:-1]),
                        'mean_50': np.mean(info['rewards'][-51:-1]),
                    }
                if args.need_probe:
                    record['D-mid'] = D_mid
                    record['h'] = h
                if len(info['rewards']) > 2:
                    record['last_reward'] = info['rewards'][-2]
                if args.eval:
                    record['eval_reward'] = np.mean(eval_info['rewards'][-args.eval_times-1:-1])

                with open(file_name + '_record.txt', 'a') as f:
                    print(record, file=f)

            # show log information only if one episode is end
            if done and len(info['rewards']) > train_episodes:
                train_episodes += 1
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))        

                #TODO
                logger.record_tabular("tau", tau)      
                logger.record_tabular("h_tau", h_tau)      
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

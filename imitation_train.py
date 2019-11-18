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
from baselines.EERL.replay_buffer import SimplestReplayBuffer
from baselines.common.misc_util import (
    set_global_seeds,
    SimpleMonitor
)

# when updating this to non-deperecated ones, it is important to
# copy over LazyFrames
from baselines.EERL.env_init import wrap_dqn_bnn
from model import binary_model_2

def parse_args():
    parser = argparse.ArgumentParser("DQN experiments for Atari games")
    # Environment
    parser.add_argument("--env", type=str, default="Pong", help="name of the game")
    parser.add_argument("--seed", type=int, default=0, help="which seed to use")
    # The number of ensemble neural networks
    parser.add_argument("--en", type=int, default=4, help="how many BNN network we need")

    # Core DQN parameters
    parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--num-steps", type=int, default=int(2e8), help="total number of steps to run the environment for")
    parser.add_argument("--batch-size", type=int, default=50, help="number of transitions to optimize at the same time")
    parser.add_argument("--learning-freq", type=int, default=4, help="number of iterations between every optimization step")
    
    # BNN evaluation setting
    parser.add_argument("--eval", action='store_true', help="whether to evaluate model")
    parser.add_argument("--eval-freq", type=int, default=1e5, help="after which step evaluate the BNN once")
    parser.add_argument("--eval-times", type=int, default=30, help="how many times evaluate the network in one evaluation time")
    parser.add_argument("--eval-expert", action='store_true', help="whether to use random actions")
    
    parser.add_argument("--alpha", type=float, default=1, help="softmax parameter")
    parser.add_argument("--bnn-explore", type=float, default=0.1, help="bnn exploration parameter")
    
    parser.add_argument("--norm", action='store_true', help="whether to norm target advantage value")
    parser.add_argument("--use-sign", action='store_true', help="whether to use sign")
    return parser.parse_args()

def make_env(game_name, noop_max):
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)  # puts rewards and number of steps in info, before environment is wrapped
    env = wrap_dqn_bnn(monitored_env, noop_max)  # applies a bunch of modification to simplify the observation space (downsample, make b/w)
    return env, monitored_env

if __name__ == '__main__':
    args = parse_args()

    print(vars(args))
    
    env_name = args.env
    seed = args.seed

    print('Game name is ', env_name)

    if args.eval:
        file_name = 'eval_en={}_seed={}_{}'.format(args.en, args.seed, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    else:
        file_name = 'en={}_seed={}_{}'.format(args.en, args.seed, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    
    if env_name == 'Qbert' or env_name == 'Breakout' or args.norm:
        file_name = 'norm_' + file_name
        
    file_name = './IL/' + env_name + '/' + file_name


    load_model_path = '../Expert_Models/{}/Saver'.format(env_name)

    if not os.path.exists(load_model_path):
        print(load_model_path, 'does not exists')
        raise NotADirectoryError

    replay_buffer = SimplestReplayBuffer(args.replay_buffer_size)

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

    #TODO    
    # records setting:
    record_set = {
        'accu-list': deque(maxlen=500),
        'sample-accu-list': [],
        'loss-list': [],
        'eval-rewards': [],
    }
    
    for i in range(args.en):
        record_set['sample-accu-list'].append(deque(maxlen=200))
        record_set['eval-rewards'].append(deque(maxlen=200))
        record_set['loss-list'].append(deque(maxlen=200))
    ############

    with U.make_session(4) as sess:
        # restore session from saver and get the expert
        saver = tf.train.import_meta_graph(load_model_path + '/Model.meta')
        saver.restore(sess, tf.train.latest_checkpoint(load_model_path))

        graph = tf.get_default_graph()

        target_input = graph.get_tensor_by_name('deepq/observation:0')        
        target_q = graph.get_tensor_by_name('deepq/q_func/add:0')
        # center
        target_a = target_q - tf.expand_dims(tf.reduce_mean(target_q, axis=1), axis=1)
        # scale
        if env_name == 'Qbert' or env_name == 'Breakout' or args.norm:
            for i in range(100):
                print('We will norm the target advantage value')
            target_a /= tf.expand_dims(tf.reduce_sum(tf.abs(target_a), axis=1), axis=1)
        
        # Create training graph and replay buffer
        bnn_act, train = EERL.imit_build_train(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            bnn_func=binary_model_2,
            learning_rate=args.lr,
            num_actions=env.action_space.n,
            en=args.en,

            raw_input_ph=target_input,
            target_output=tf.stop_gradient(target_a),

            gamma=0.99,
            grad_norm_clipping=10,
            alpha=args.alpha,
            bnn_explore=args.bnn_explore,

            use_sign=args.use_sign
        )
 
        # initialization
        # To avoid the expert's variables' scope name conflict with our new variables' scope name
        new_vars = [var for var in tf.global_variables() if 'Imitation' in var.name]
        tf.get_default_session().run(tf.variables_initializer(new_vars))
        
        obs = env.reset()
        
        last_done_steps = 0
        last_death_steps = 0

        if not os.path.exists('./IL/' + env_name):
            os.makedirs('./IL/' + env_name)

        #TODO.
        with open(file_name + '_record.txt', 'a') as f:
            print(sys.executable + " " + " ".join(sys.argv), file=f)
            print(vars(args), file=f)

        num_iters = 0
        sample_size = args.en * args.batch_size

        #TODO
        accu_list = deque(maxlen=500)

        #TODO
        if args.eval_expert:
            eval_obs = eval_env.reset()
            eval_done = False
            target_label = tf.squeeze(tf.argmax(target_a, axis=1))
            for i in range(100000):
                eval_act, eval_adv = sess.run([target_label, target_a],
                        feed_dict={target_input:np.array(eval_obs)[None]})
                eval_obs, eval_r, eval_done, eval_info = eval_env.step(eval_act)
                if eval_done:
                    eval_obs = eval_env.reset()
                    print(eval_info)

                if i % 2000 == 0:
                    print(eval_act, eval_adv)

        while True:            
            num_iters += 1
            
            replay_buffer.add(obs)

            # Take action and store transition in the replay buffer.
            action, act_value = bnn_act(np.array(obs)[None])
            
            if num_iters % 1000 == 0:
                expert_a, expert_label = sess.run([target_a, tf.argmax(tf.squeeze(target_a))], {target_input:np.array(obs)[None]})
                print(expert_a, np.mean(expert_a))
                print(act_value, np.mean(act_value))
                if expert_label == action:
                    accu_list.append(1)
                else:
                    accu_list.append(0)


            new_obs, rew, done, info = env.step(action)       
            obs = new_obs
            
            if done:
                obs = env.reset()

            if num_iters > max(5 * args.batch_size, args.replay_buffer_size // 20) and num_iters % args.learning_freq == 0:
            
                obses_t = replay_buffer.sample(sample_size)
                outputs = train(obses_t)

                #TODO                
                for i in range(args.en):
                    record_set['sample-accu-list'][i].append(outputs[i])
                    record_set['loss-list'][i].append(outputs[args.en + i])

            if info["steps"] > args.num_steps:
                break

            #TODO
            if num_iters % 5000 == 0:
                record = {
                        'steps': info["steps"],
                        'mean_100': np.mean(info['rewards'][-101:-1]),
                        'mean_50': np.mean(info['rewards'][-51:-1]),
                        'mean_20': np.mean(info['rewards'][-21:-1]),
                    }
                if len(info['rewards']) > 2:
                    record['last_reward'] = info['rewards'][-2]
                
                if args.eval:
                    record['eval_reward'] = eval_score

                with open(file_name + '_record.txt', 'a') as f:                    
                    print(record, file=f)

            if done and len(info['rewards']) > train_episodes:
                train_episodes += 1
                logger.record_tabular("steps", info["steps"])
                logger.record_tabular("iters", num_iters)
                logger.record_tabular("episodes", len(info["rewards"]))         
                
                mean_50_r = np.mean(info["rewards"][-51:-1])
                mean_20_r = np.mean(info["rewards"][-21:-1])
                logger.record_tabular("recent reward (20 epi mean)", mean_20_r)
                logger.record_tabular("recent reward (50 epi mean)", mean_50_r)
                logger.record_tabular("reward (100 epi mean)", np.mean(info["rewards"][-101:-1]))

                #TODO.
                if len(info['rewards']) > 2:
                    logger.record_tabular("last reward", info["rewards"][-2])    
                for i in range(args.en):                    
                    logger.record_tabular("No.{} sample accuracy is ".format(i), np.mean(record_set['sample-accu-list'][i]))
                logger.record_tabular("execute sample accuracy is ", np.mean(accu_list))
                
                if args.eval:
                    logger.record_tabular("evaluation score", eval_score)
                    
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
                    
                    eval_act = bnn_act(np.array(eval_obs)[None])[0]
                    eval_obs, r, eval_done, eval_info = eval_env.step(eval_act)
                print(eval_info['rewards'][history_episodes_num:])

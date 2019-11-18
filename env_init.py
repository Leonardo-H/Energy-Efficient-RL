from baselines.common.atari_wrappers_deprecated import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ProcessFrame84, FrameStack, ClippedRewardsWrapper

def wrap_dqn_bnn(env, noop_max=1):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    
    for i in range(10):
        print('Clip rewards')
    env = ClippedRewardsWrapper(env)
    return env

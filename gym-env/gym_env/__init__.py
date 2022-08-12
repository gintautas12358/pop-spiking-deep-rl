from gym.envs.registration import register 

register(
    id='PegInHole-v0',
    entry_point='gym_env.envs:PegInHole',
) 

register(
    id='PegInHole-rand',
    entry_point='gym_env.envs:PegInHoleRandom',
) 

register(
    id='PegInHole-rand_events',
    entry_point='gym_env.envs:PegInHoleRandomEvents',
) 

register(
    id='PegInHole-rand_events_depth',
    entry_point='gym_env.envs:PegInHoleRandomEventsDepth',
) 



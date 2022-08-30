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

register(
    id='PegInHole-rand_events_depth_activity_center',
    entry_point='gym_env.envs:PegInHoleRandomEventsDepthActivityCenter',
) 

register(
    id='PegInHole-rand_events_visual_servoing',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoing',
) 


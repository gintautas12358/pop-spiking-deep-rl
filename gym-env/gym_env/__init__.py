from gym.envs.registration import register 



register(
    id='PegInHole-test',
    entry_point='gym_env.envs:PegInHole_test',
) 

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

register(
    id='PegInHole-rand_events_visual_servoing_guiding',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuiding',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_activity',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingActivity',
) 


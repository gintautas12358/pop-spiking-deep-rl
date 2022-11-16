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
    id='PegInHole-rand_events_visual_servoing_guiding2',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuiding2',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding2_no_coord',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuiding2NC',
) 




register(
    id='PegInHole-rand_events_visual_servoing_guiding2_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuiding2Rand',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding2_no_coord_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuiding2NCRand',
) 







register(
    id='PegInHole-rand_events_visual_servoing_guiding_activity',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingActivity',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_activity2',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingActivity2',
)

register(
    id='PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingActivity2NC',
)

register(
    id='PegInHole-rand_events_visual_servoing_guiding_activity2_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingActivity2Rand',
)

register(
    id='PegInHole-rand_events_visual_servoing_guiding_activity2_no_coord_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingActivity2NCRand',
)

register(
    id='PegInHole-rand_events_visual_servoing_guiding_corner_activity',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingCornerActivity',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_corner_activity2',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingCornerActivity2',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingCornerActivity2NC',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_corner_activity2_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingCornerActivity2Rand',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_corner_activity2_no_coord_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingCornerActivity2NCRand',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_vae',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingVAE',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_vae2',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingVAE2',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingVAE2NC',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_vae2_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingVAE2Rand',
) 

register(
    id='PegInHole-rand_events_visual_servoing_guiding_vae2_no_coord_rand',
    entry_point='gym_env.envs:PegInHoleRandomEventsVisualServoingGuidingVAE2NCRand',
) 
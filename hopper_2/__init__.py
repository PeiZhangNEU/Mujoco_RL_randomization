from gym.envs.registration import register

register(
    id='HopperRandom-v1',
    entry_point='hopper_2.envs:Hopper2',
    max_episode_steps=200,
)

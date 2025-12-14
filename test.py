import os
from importlib.metadata import version
import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper

# Print version
# print("robosuite version:", version("robosuite"))

# Fix RoboSuite logs for HPC
os.environ["ROBOSUITE_LOG_PATH"] = "/scratch/gilbreth/athani/reinforcement_learning/robosuite/robosuite_logs"
os.makedirs(os.environ["ROBOSUITE_LOG_PATH"], exist_ok=True)

# Correct controller loader for robosuite 1.5.1
from robosuite import load_composite_controller_config
# from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

if __name__ == '__main__':

    # Load composite controller (BASIC is recommended)
    controller_configs = load_composite_controller_config(
        controller="BASIC"
    )

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],  # use Panda robot. has only right arm.
        controller_configs=controller_configs,
        has_renderer=True,
        render_camera = 'frontview',
        has_offscreen_renderer = True,
        use_camera_obs = 'False', 
        reward_shaping=True, # step by step improvements
        horizon=300, # epsiode length. since these enviroments dont have a terminal state.
        control_freq=20, # 20 hz of action

    )

    # Wrap with Gymnasium wrapper

    env = GymWrapper(env)





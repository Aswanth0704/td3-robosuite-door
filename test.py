import os
from importlib.metadata import version
import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent


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

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    tau = 0.005

    agent = Agent(actor_learning_rate, critic_learning_rate, input_dims = env.observation_space.shape, tau = tau, env = env, gamma = 0.99, update_actor_interval = 2,
                    warmup = 1000, n_actions = env.action_space.shape[0], max_size = 100000, layer1_size = layer1_size, layer2_size = layer2_size, batch_size = batch_size, noise = 0.1)
    n_games = 3
   

    epsiode_identifier = f'1-actor_lr={actor_learning_rate}-critic_lr={critic_learning_rate}-batch_size={batch_size}-CriticAdamW-tau={tau}-l1={layer1_size}-l2={layer2_size}'
    agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset(seed)
        done = False
        # truncated = False
        score = 0
        while (not done):
            action = agent.choose_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            score+= reward
            # Move to the next state:
            done = terminated or truncated
            observation = new_observation


        print(f"Episode {i} | Score: {score:.2f}")




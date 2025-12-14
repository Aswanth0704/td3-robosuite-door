import os
import torch as T
import numpy as np
from importlib.metadata import version
import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from td3_torch import Agent
from torch.utils.tensorboard import SummaryWriter

seed = 42
np.random.seed(seed)
T.manual_seed(seed)



# Print version
# print("robosuite version:", version("robosuite"))

# Fix RoboSuite logs for HPC
os.environ["ROBOSUITE_LOG_PATH"] = "/scratch/gilbreth/athani/reinforcement_learning/robosuite/robosuite_logs"
os.makedirs(os.environ["ROBOSUITE_LOG_PATH"], exist_ok=True)

# Correct controller loader for robosuite 1.5.1
from robosuite import load_composite_controller_config
# from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

if __name__ == '__main__':

    # Load composite controller (BASIC is recommended). Try different configs!
    controller_configs = load_composite_controller_config(
        controller="BASIC"
    )

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],  # use Panda robot. has only right arm.
        controller_configs=controller_configs,
        has_renderer=False,
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

    writer = SummaryWriter('logs')
    n_games = 10000
    best_score = 0
    score_history = []
    load_checkpoint = False

    epsiode_identifier = f'0-actor_lr={actor_learning_rate}-critic_lr={critic_learning_rate}-batch_size={batch_size}-CriticAdamW-tau={tau}-l1={layer1_size}-l2={layer2_size}'
    models_loaded = agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset(seed)
        done = False
        truncated = False
        score = 0
        while (not done) and (not truncated):
            action = agent.choose_action(observation)
            new_observation, reward, done, truncated, info = env.step(action)
            score+= reward
            agent.remember(observation, action, reward, new_observation, done)
            # may be I am trying to learn so fast here:
            # something like this is probably more efficient
            if agent.time_step > agent.warmup:
                agent.learn()
            agent.learn()

            observation = new_observation

        if(len(score_history)>100):
            avg_score = np.mean(score_history[-100:])
        else:
            avg_score = np.mean(score_history)

        score_history.append(score)
        writer.add_scalar(f"score - {epsiode_identifier}", score, global_step = i)
        writer.add_scalar(f"avg_score - {epsiode_identifier}", avg_score, global_step = i)

        if(avg_score > best_score):
            best_score = avg_score
        if(i%10 == 0):
            agent.save_models()

        print(f"Episode {i} | Score: {score:.2f} | Avg: {avg_score:.2f}")









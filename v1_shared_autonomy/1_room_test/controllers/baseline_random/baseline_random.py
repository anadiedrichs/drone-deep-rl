"""
Test a laggy base-pilot in room 1
"""
import sys
import os
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.monitor import Monitor
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *
import numpy as np


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    max_episode_steps=10_000
    model_total_timesteps = 100_000
    log_path = "./20250122/"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "baseline-SimpleCornerEnvRS10.csv")

def save_results(r,l,g,file_name):
    data = {'Reward': r, 'Len': l, 'Goal':g}
    # Create DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False,mode='w')

def run_experiment(want_to_train=True):
    args = Params()
    # Initialize the environment
    #env = CornerEnv()
    env = SimpleCornerEnvRS10(args.max_episode_steps)
    env.set_trajectory_path(args.log_path)
    env = Monitor(env, filename=args.log_path, info_keywords=env.get_info_keywords())
    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    goal=[]
    steps = 0
    # https://github.com/openai/gym/issues/681
    # env.seed(args.model_seed)
    obs, _ = env.reset(seed=args.model_seed)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    save_results(reward_sum, len_total, goal, args.eval_result_file)

    while True:
        action = env.action_space.sample()
        # return obs, reward, done, self.truncated, info
        obs, reward, done, truncated, info = env.step(action)
        # print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward
        # if an episode ends
        if done or args.model_total_timesteps < steps:
            if done and not truncated:
                goal.append(True)
                print("Goal reached! rew = ", reward_sum)
                print("Goal reached! len = ", steps)
            else:
                goal.append(False)

            # total reward per episode
            reward_array.append(reward_sum)
            # total steps taken per episode
            len_total.append(steps)
            save_results(reward_array,len_total,goal,args.eval_result_file)
            # new episode
            obs = env.reset()
            total += 1
            # reset variables
            reward_sum = 0
            steps = 0

        if total == args.n_eval_episodes:
            break

    # Create DataFrame
    df = pd.DataFrame({'Reward': reward_array, 'Len': len_total, 'Goal':goal})
    print("Results")
    print(df)

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")


if __name__ == '__main__':

    run_experiment()

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
    model_total_timesteps = 50_000
    log_path = "./logs_2024-10-07/"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "baseline-random-results.csv")

def save_results(r,l,g,file_name):
    data = {'Reward': r, 'Len': l, 'Goal':g}
    # Create DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False,mode='w')

def run_experiment(want_to_train=True):
    args = Params()
    # Initialize the environment
    env = CornerEnv()
    env = Monitor(env, filename=args.log_path, info_keywords=("is_success","corner",))
    # https://github.com/openai/gym/issues/681
    env.action_space.np_random.seed(args.model_seed)

    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    goal=[]
    steps = 0
    obs = env.reset()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    save_results(reward_sum, len_total, goal, args.eval_result_file)

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward
        # if an episode ends
        if done or args.model_total_timesteps < steps:
            if done:
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

"""
Test a laggy pilot in room 1
"""

import sys
import os
from controller import Supervisor
sys.path.append('../../../../utils')
from utilities import *
from pid_controller import *

sys.path.append('../../../../pilots')
from NoisyPilot import *

sys.path.append('../../../../copilot')
from CrazyflieDrone import *

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
import pandas as pd


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class StopExperimentCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(StopExperimentCallback, self).__init__(verbose)

    def _on_step(self):
        # print("ON STEP")
        # Access the environment from the model
        terminated = self.model.env.envs[
            0].terminated  # Assumes that the first environment has the termination attribute

        if terminated:  # done is True, drone reach the corner
            self.logger.info("The drone reached the corner !! :-)")
            self.logger.record("is_success", 1)

        # print(self.model.env.envs[0].episode_score)

        cumm_score = self.model.env.envs[0].episode_score

        # analyze reward threshold
        if cumm_score <= -1_000_000 or cumm_score > 1_000_000:
            self.model.env.envs[0].truncated = True
            self.model.env.envs[0].is_success = False
            self.logger.record("is_success", 0)
            return False  # stop the training

        return True  # Return whether the training stops or not


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    model_total_timesteps = 100_000
    model_verbose = True
    kl_target = 0.015
    #gae_gamma = 0.99
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 512
    ls_rate = 0.001  # linear schedule rate
    log_path = "./logs/"
    save_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"  # os.path.join(log_path,"ppo_model_pilot_room_1")
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")


def run_experiment(want_to_train=True):
    args = Params()
    # Initialize the environment
    env = CornerEnv()
    # env = Monitor(env, filename=args.log_path, info_keywords=("is_success",))

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps, verbose=args.model_verbose,
                target_kl=args.kl_target,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=linear_schedule(args.ls_rate),
                seed=args.model_seed, tensorboard_log=args.log_path)
    # set up logger
    # new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    # model.set_logger(new_logger)

    # evaluate the agent
    # Load a saved model
    model.load(args.save_model_path)

    # Evaluate the policy
    pilot = NoisyPilot(model, args.model_seed)
    obs = env.reset()
    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    steps = 0

    while True:
        # print("obs=", obs)
        action, _ = pilot.choose_action(obs)
        # print("Action: ", action)
        obs, reward, done, info = env.step(action)
        # print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward

        if done:
            print("Goal reached! rew = ", reward_sum)
            print("Goal reached! len = ", steps)
            # total reward per episode
            reward_array.append(reward_sum)
            # total steps taken per episode
            len_total.append(steps)
            # new episode
            obs = env.reset()
            total += 1
            # reset variables
            reward_sum = 0
            steps = 0

        if total == args.n_eval_episodes:
            break

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    data = {'Reward': reward_array, 'Len': len_total}
    # Create DataFrame
    df = pd.DataFrame(data)
    print("Results")
    print(df)
    # Save DataFrame to CSV
    df.to_csv(args.eval_result_file, index=False)

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")


if __name__ == '__main__':

    run_experiment()

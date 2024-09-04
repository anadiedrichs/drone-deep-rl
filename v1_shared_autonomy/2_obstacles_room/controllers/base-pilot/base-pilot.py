"""

Create a custom environment.

Then we test the different methods

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils import *
from copilot.CrazyflieDrone import TargetAndObstaclesEnv
from pilots import *
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
        # Assumes that the first environment has the termination attribute
        terminated = self.model.env.envs[0].terminated

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
    model_total_timesteps = 200_000
    model_verbose = True
    kl_target = 0.015
    # gae_gamma = 0.99
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 512
    # linear schedule rate
    ls_rate = 0.001
    log_path = "./logs/"
    save_model_path = os.path.join(log_path, "ppo_model_pilot_room_2")
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")


def run_experiment(want_to_train: bool) -> object:
    args = Params()
    # TODO train with different targets: cone_1, cone_2 and cone_3
    # Initialize the environment
    env = TargetAndObstaclesEnv("cone_3")
    env = Monitor(env, filename=args.log_path, info_keywords=("is_success",))

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps, verbose=args.model_verbose,
                target_kl=args.kl_target,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=linear_schedule(args.ls_rate),
                seed=args.model_seed, tensorboard_log=args.log_path)
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        custom_callback = StopExperimentCallback()
        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback,
                    progress_bar=True)
        # Save the learned model
        model.save(args.save_model_path)

    else:
        # evaluate the agent
        # Load a saved model
        model.load(args.save_model_path)

        obs = env.reset()
        # Evaluate the policy
        mean_reward, std_reward = evaluate_policy(model, env,
                                                  n_eval_episodes=args.n_eval_episodes,
                                                  deterministic=True,
                                                  return_episode_rewards=True)
        # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

        data = {'Reward': mean_reward, 'Len': std_reward}
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
    run_experiment(want_to_train=True)

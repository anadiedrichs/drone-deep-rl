"""

Create a custom environment.

Then we test the different methods

"""

import sys
import os
sys.path.append('../../utils')
from utilities import *
from pid_controller import *

sys.path.append('../../copilot')
from CrazyflieDrone import *
from Copilots import *
sys.path.append('../../pilots')
from pilot import *
from LaggyPilot import *
from NoisyPilot import *
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
        terminated = self.model.env.envs[0].terminated  # Assumes that the first environment has the termination attribute

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
            return False # stop the training

        return True # Return whether the training stops or not

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
    ls_rate = 0.001 # linear schedule rate
    log_path = "./logs/"
    pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    save_model_path = "./logs/ppo_model_copilot_laggy_room_1" # os.path.join(log_path,"ppo_model_pilot_room_1")
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "laggy-copilot-results.csv")
    copilot_alpha = 0.5 # copilot threshold


def run_experiment(want_to_train=True, pilot:Pilot):

    args: Params = Params()

    # Initialize the environment
    env = CopilotCornerEnv()

    # Load the pilot model

    # set pilot
    env.set_pilot(pilot, args.copilot_alpha)
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
    env.training_mode = want_to_train

    # Evaluate the policy
    if want_to_train:
        custom_callback = StopExperimentCallback()
        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback)
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


def train_copilot(seed:None, alpha:None, is_pilot_noisy:True):
    args = Params()

    args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    pilot_model = PPO.load(args.pilot_model_path)

    if alpha is not None:
        args.copilot_alpha = alpha
    if seed is not None:
        args.model_seed = seed

    if is_pilot_noisy:
        args.log_path = "./logs-noisy/"
        args.save_model_path = os.path.join(args.log_path, "model_noisy_copilot_room_1")
        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(pilot_model, args.model_seed)
    else:
        args.log_path = "./logs-laggy/"
        args.save_model_path = os.path.join(args.log_path, "model_laggy_copilot_room_1")
        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(pilot_model, args.model_seed)


    run_experiment(True, pilot)

def test_copilot(seed:None, alpha:None, is_pilot_noisy:True):
    args = Params()

    args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    pilot_model = PPO.load(args.pilot_model_path)

    if alpha is not None:
        args.copilot_alpha = alpha
    if seed is not None:
        args.model_seed = seed

    if is_pilot_noisy:
        # where is the model to load
        args.save_model_path = "./logs-noisy/model_noisy_copilot_room_1"
        args.log_path = "./logs-noisy_test/"
        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(pilot_model, args.model_seed)
    else:
        args.save_model_path = "./logs-laggy/model_laggy_copilot_room_1"
        args.log_path = "./logs-laggy_test/"
        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(pilot_model, args.model_seed)


    run_experiment(False, pilot)

if __name__ == '__main__':
    # train noisy copilot
    # train_copilot(seed=5,alpha=0.5,True)
    # train laggy copilot
    # train_copilot(seed=5,alpha=0.5,False)

    # test noisy copilot
    # test_copilot(seed=5,alpha=0.5,True)
    # train laggy copilot
    # test_copilot(seed=5,alpha=0.5,False)
"""

Create a custom environment.

Then we test the different methods

"""

import sys
from controller import Supervisor

sys.path.append('../../../../utils')
from utilities import *
from pid_controller import *

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
    def __init__(self, verbose=0):
        super(StopExperimentCallback, self).__init__(verbose)

    def _on_step(self):
        # print("ON STEP")
        # Access the environment from the model
        terminated = self.model.env.envs[0].terminated  # Assumes that the first environment has the termination attribute

        if terminated:  # done is True, drone reach the corner
            self.model.logger.info("The drone reached the corner !! :-)")
            self.logger.record("is_success", 1)

        # print(self.model.env.envs[0].episode_score)

        cumm_score = self.model.env.envs[0].episode_score

        # analyze reward threshold
        if cumm_score <= -1_000_000 or cumm_score > 1_000_000:
            self.model.env.envs[0].truncated = True
            self.model.env.envs[0].is_success = False
            self.logger.record("is_success", 0)
            return False # stop the training

        return True # Return whether the trainig stops or not

def train():
    # Initialize the environment
    env = CornerEnv()
    #env = DummyVecEnv([lambda: env])
    tmp_path = "./logs-2024-08-04/"
    env = Monitor(env, filename=tmp_path, info_keywords=("is_success",))


    # model to train: PPO
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1, seed=5, tensorboard_log=tmp_path)
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)
    # access to tensorboard logs typing in your terminal:
    # tensorboard --logdir ./logs/


    # Stops training when the model reaches the maximum number of episodes
    # callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=50, verbose=1)
    # Create the callback list
    # callback_list = CallbackList([eval_callback, callback_max_episodes])

    # Almost infinite number of timesteps, but the training will stop early
    # as soon as the number of consecutive evaluations without model
    # improvement is greater than 3

    custom_callback = StopExperimentCallback()
    # start training
    print("INICIO ENTRENAMIENTO")
    model.learn(total_timesteps=50_000, callback=custom_callback,
                progress_bar=True)

    # Save the learned model
    model.save("./logs-2024-08-04/ppo_model_pilot_room_1")


    # Test the trained agent
    # using the env
    obs = env.reset()
    n_steps = 200
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)

        if done:
            # Note that the env resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break

    print("EXPERIMENT ended")
    # close Webots simulator
    # env.simulationQuit(0)

def evaluate():
    env = CornerEnv()
    # env = DummyVecEnv([lambda: env])
    tmp_path = "./logs_evaluate/"
    env = Monitor(env, filename=tmp_path, info_keywords=("is_success",))

    # model to train: PPO
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1, seed=5, tensorboard_log=tmp_path)
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    # Load a saved model
    model.load("./logs_2028_08_02/ppo_model_pilot_room_1")

    obs = env.reset()
    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env,
                                              n_eval_episodes=2,
                                              deterministic=True,
                                              return_episode_rewards=True)

    #print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    print("Mean reward")
    print(mean_reward)
    print("std_reward")
    print(std_reward)
    print("EXPERIMENT ended")
    # close Webots simulator
    # env.simulationQuit(0)

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
    batch_size = 512
    n_steps = 128
    lr_rate = linear_schedule(0.01)
    log_path = "./logs-2024-08-04/"
    save_model_path = "./logs-2024-08-04/ppo_model_pilot_room_1"
    # increase this number later
    n_eval_episodes = 2

def run_experiment(want_to_train=True):

    args = Params()
    # Initialize the environment
    env = CornerEnv()
    env = Monitor(env, filename=args.log_path, info_keywords=("is_success",))

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps, verbose=args.model_verbose,
                target_kl=args.kl_target,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=args.lr_rate,
                seed=args.model_seed, tensorboard_log=args.log_path)
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        custom_callback = StopExperimentCallback()
        # start training
        print("INICIO ENTRENAMIENTO")
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback,
                    progress_bar=True)

        # Save the learned model
        model.save(args.save_model_path)

        print("FIN ENTRENAMIENTO")

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
        print("Mean reward")
        print(mean_reward)
        print("std_reward")
        print(mean_reward)

        data = {'Reward': mean_reward, 'Len': mean_reward}
        # Create DataFrame
        df = pd.DataFrame(data)
        # Print the DataFrame
        print(df)
        # Save DataFrame to CSV
        df.to_csv('results.csv', index=False)

        print("Evaluation ended")
        # close Webots simulator
        # env.simulationQuit(0)



if __name__ == '__main__':
    # train()
    # evaluate()
    run_experiment(True)
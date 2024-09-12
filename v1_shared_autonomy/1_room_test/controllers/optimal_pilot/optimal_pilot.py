"""
Script to test and train a base / optimal pilot in room 1
"""

import sys
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *
from datetime import datetime

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
        if cumm_score <= -100_000 or cumm_score > 100_000:
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
    model_total_timesteps = 40_000
    model_verbose = True
    kl_target = 0.015
    gae_gamma = 0.98
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 2048
    lr_rate = 0.0001
    # Cambia la arquitectura de la red a dos capas de 32 neuronas cada una
    policy_kwargs = dict(
        net_arch=[20, 20]
    )
    ls_rate = 0.00001  # linear schedule rate
    ent_coef=0.05
    log_path = "./logs-2024-09-11_2345/"
    save_model_path = "./logs-2024-09-11_2345/ppo_model_pilot_room_1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")


def run_experiment(want_to_train=True):
    start_time = datetime.now()
    args = Params()
    # Initialize the environment
    env = CornerEnv()
    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps,
                verbose=args.model_verbose,
                target_kl=args.kl_target,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=args.lr_rate,#linear_schedule(args.ls_rate),
                policy_kwargs=args.policy_kwargs,
                ent_coef=args.ent_coef,
                seed=args.model_seed,
                tensorboard_log=args.log_path)
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([StopExperimentCallback(), HParamCallback()])

        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback_list,
                    progress_bar=True)
        # Save the learned model
        model.save(args.save_model_path)

    else:

        # Load a saved model
        print("TESTING MODE ")
        model.load(args.save_model_path)
        obs = env.reset()
        # Evaluate the agent policy
        mean_reward, std_reward = evaluate_policy(model, env,
                                                  n_eval_episodes=args.n_eval_episodes,
                                                  #deterministic=True,
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
    end_time = datetime.now()
    # Calcular la duración
    duration = end_time - start_time
    save_experiment_time(start_time,end_time,args.log_path)

if __name__ == '__main__':

    run_experiment(want_to_train=True)


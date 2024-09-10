"""
Custom environment for base pilot or optimal pilot in room 1
"""

import sys
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CrazyflieDrone import CornerEnv
from pilots import *

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            # TODO how to save ls_rate
            #lr is a function
            #"learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda":self.model.gae_lambda,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "seed": self.model.seed,
            "target_kl": self.model.target_kl,
            "ent_coef":self.model.ent_coef,
            "total_timesteps":self.model._total_timesteps,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean":0.0,
            "train/value_loss": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


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
    model_total_timesteps = 80_000
    model_verbose = True
    kl_target = 0.015
    #gae_gamma = 0.99
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 1024
    ls_rate = 0.001  # linear schedule rate
    ent_coef=0.05
    log_path = "./logs-2024-09-10_ent_coef/"
    save_model_path = "./logs-2024-09-10_ent_coef/ppo_model_pilot_room_1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")


def run_experiment(want_to_train=True):
    args = Params()
    # Initialize the environment
    env = CornerEnv()
    env = Monitor(env, filename=args.log_path, info_keywords=("is_success","corner",))

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps, verbose=args.model_verbose,
                target_kl=args.kl_target,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=linear_schedule(args.ls_rate),
                ent_coef=args.ent_coef,
                seed=args.model_seed, tensorboard_log=args.log_path)
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
        # evaluate the agent
        # Load a saved model
        print("TESTING MODE ")
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

if __name__ == '__main__':
    # train()
    # evaluate()
    run_experiment(want_to_train=True)

"""
Script to train / test a pilot for room 2
"""
import sys
import os
from time import strftime
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.TargetObstacleEnv import *
from pilots import *
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
# tensorboard hyperparameter logging
class HParamCallback1(BaseCallback):
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
            "ent_coef": self.model.ent_coef,
            "total_timesteps": self.model._total_timesteps,
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
        # Assumes that the first environment has the termination attribute
        terminated = self.model.env.envs[0].terminated
        # done is True
        if terminated:
            self.logger.info("TERMINATED, SCORE ", self.model.env.envs[0].episode_score,exclude=("json", "csv"))
            self.logger.record("target_name",self.model.env.envs[0].target_name,exclude=("json", "csv"))

        cumm_score = self.model.env.envs[0].episode_score
        # analyze reward threshold
        if cumm_score <= -100_000:
            self.model.env.envs[0].truncated = True
            self.model.env.envs[0].is_success = False
            self.logger.record("is_success", 0)
            self.logger.record("target_name",self.model.env.envs[0].target_name)
            return False  # stop the training

        return True  # Return whether the training stops or not


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    model_total_timesteps = 60_000
    model_verbose = True
    kl_target = 0.013
    gamma = 0.97
    ent_coef = 0.05
    gae_gamma = 0.98
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 2048
    # linear schedule rate
    lr_rate = 0.00001
    ls_rate = 0.001
    log_path = "./logs_2024-10-14/"
    monitor_file_name = os.path.join(log_path, strftime("%Y-%m-%d_%H%M%S")+"-monitor.csv")
    save_model_path = os.path.join(log_path, strftime("%Y-%m-%d_%H%M%S")+"_ppo_model_pilot_room_2")
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, strftime("%Y-%m-%d_%H%M%S")+"-results.csv")

def run_experiment(want_to_train: bool) -> object:
    args = Params()
    # Initialize the environment
    env = TargetAndObstaclesEnv(seed=args.model_seed)
    env = Monitor(env, filename=args.monitor_file_name,
                  info_keywords=get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps,
                verbose=args.model_verbose,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=args.lr_rate,
                ent_coef=args.ent_coef,
                seed=args.model_seed,
                tensorboard_log=args.log_path)
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        # Create the callback list
        custom_callback_list = CallbackList([StopExperimentCallback(), HParamCallback1()])
        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback_list,
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
                                                  deterministic=False,
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

    run_experiment(want_to_train=True)

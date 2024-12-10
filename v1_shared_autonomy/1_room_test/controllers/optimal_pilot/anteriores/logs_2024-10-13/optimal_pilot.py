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
from stable_baselines3.common.monitor import Monitor


class StopExperimentCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(StopExperimentCallback, self).__init__(verbose)

    def _on_step(self):
        # print("ON STEP")
        cumm_score = self.model.env.envs[0].episode_score
        # analyze reward threshold
        if cumm_score <= -100_000 or cumm_score > 100_000:
            self.model.env.envs[0].truncated = True
            self.model.env.envs[0].is_success = False
            self.logger.record("is_success", 0)
            return False  # stop the training

        # Access the environment from the model
        # Assumes that the first environment has the termination attribute
        terminated = self.model.env.envs[0].terminated
        if terminated:  # done is True, drone reach the corner
            self.logger.info("The drone reached the corner !! :-)")
            self.logger.record("is_success", 1)


        return True  # the training continues


class DefaultParams:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    model_total_timesteps = 40_000
    model_verbose = True
    log_path = "./logs_2024-10-13_test/"
    save_model_path = "./logs_2024-10-13/ppo_model_pilot_room_1_256"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "2024-10-13_256_test"


class Params1(DefaultParams):
    """
    Common model hyperparameters for PPO and
    other configs
    """
    # Considerar más recompensas futuras
    gamma = 0.97
    batch_size = 64
    # Más pasos por actualización para mayor estabilidad
    n_steps = 2048
    # Menor tasa de aprendizaje para evitar que converja demasiado rápido
    lr_rate = 5e-5
    # Permitir mayor exploración con un rango de clipping más amplio
    clip_range = 0.2
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.05
    # Cambia la arquitectura de la red a dos capas de 32 neuronas cada una
    policy_kwargs = dict(
        net_arch=[256, 256]
    )
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    kl_target = 0.015
    gae_lambda = 0.95
    vf_coef = 0.5


def run_experiment_defaults(want_to_train=True):
    start_time = datetime.now()
    args = DefaultParams()
    # Initialize the environment
    env = CornerEnv()

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
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
                    callback=custom_callback_list)
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
    end_time = datetime.now()
    # Calcular la duración
    duration = end_time - start_time
    save_experiment_time(start_time, end_time, args.log_path)


def run_experiment_params1(want_to_train=True):
    start_time = datetime.now()
    args = Params1()
    # Initialize the environment
    env = CornerEnv()

    env.set_trajectory_file_name(args.log_path+"drone_walk.csv")
    print(env.get_trajectory_file_name())
    print("Tipee Y para confirmar correr el experimento")
    env.wait_keyboard()
    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps,
                verbose=args.model_verbose,
                #target_kl=args.kl_target,
                batch_size=args.batch_size,
                gamma=args.gamma,
                learning_rate=args.lr_rate,
                clip_range=args.clip_range,
                policy_kwargs=args.policy_kwargs,
                ent_coef=args.ent_coef,
                seed=args.model_seed,
                tensorboard_log=args.log_path
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([StopExperimentCallback(), HParamCallback()])

        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback_list,
                    tb_log_name=args.tb_log_name)
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
    end_time = datetime.now()
    # Calcular la duración
    duration = end_time - start_time
    save_experiment_time(start_time, end_time, args.log_path)


if __name__ == '__main__':
    # run_experiment_defaults(want_to_train=False)
    #run_experiment_default_noisy_obs(want_to_train=False)
    run_experiment_params1(want_to_train=False)

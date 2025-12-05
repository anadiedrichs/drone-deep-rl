"""
Script to test and train a base / optimal pilot in room 1
"""

import sys
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import os
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *


# tensorboard hyperparameter logging
class HParamCallback1(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate(1) if callable(self.model.learning_rate) else self.model.learning_rate,
            "gamma": self.model.gamma,
            #"gae_lambda": self.model.gae_lambda,
           # "n_steps": self.model.n_steps,
           # "batch_size": self.model.batch_size,
            "seed": self.model.seed,
           # "target_kl": self.model.target_kl,
           # "ent_coef": self.model.ent_coef,
           # "n_epochs": self.model.n_epochs,
           # "clip_range": self.model.clip_range(1) if callable(self.model.clip_range) else self.model.clip_range,
            "total_timesteps": self.model._total_timesteps,
        }
        # Ensure all values are compatible with TensorBoard
        hparam_dict = {k: (v if isinstance(v, (int, float, str, bool, torch.Tensor)) else str(v))
                       for k, v in hparam_dict.items()}
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0.0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json"),
        )

    def _on_step(self) -> bool:
        return True


class StopExperimentCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(StopExperimentCallback, self).__init__(verbose)
        self.episode_step = 0

    def _on_step(self):
        # print("ON STEP")
        cumm_score = self.model.env.envs[0].episode_score

        # Incrementar el contador de pasos del episodio
        self.episode_step += 1
        self.model.env.envs[0].increment_episode_step()
        # analyze reward threshold
        #if (cumm_score <= self.model.env.envs[0].MIN_EPISODE_SCORE or
        #        cumm_score > self.model.env.envs[0].MAX_EPISODE_SCORE):
        #    self.model.env.envs[0].truncated = True
        #    self.model.env.envs[0].is_success = False
        #    self.logger.record("is_success", 0)
        #    self.episode_step = 0
        #    return False  # stop the training

        # Access the environment from the model
        # Assumes that the first environment has the termination attribute
        terminated = self.model.env.envs[0].terminated
        if terminated:  # done is True, drone reach the corner
            self.logger.info("The drone reached the corner !! :-)")
            self.logger.record("is_success", 1)
            self.episode_step = 0

        return True  # the training continues


class DefaultParams:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    model_total_timesteps = 50_000
    model_verbose = True
    log_path = "./logs_2024-11_01_params2/"
    save_model_path = "./logs_2024-11_01_params2/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "2024-11-04"


class Params2:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./logs_2024-11_04_CornerEnvShaped20/"
    save_model_path = "./logs_2024-11_04_CornerEnvShaped20/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "2024-11-04"
    # Considerar más recompensas futuras
    gamma = 0.97
    #  tasa de aprendizaje
    lr_rate = 1e-4

    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[64, 64]
    )
    kl_target = 0.015
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate

    gae_lambda = 0.95
    vf_coef = 0.5
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.05


def run_experiment_CornerEnvShaped20(want_to_train=True):
    start_time = datetime.now()
    args = Params2()
    if want_to_train is False:
        args.log_path += "test/"
    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = CornerEnvShaped20(args.model_total_timesteps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                target_kl=args.kl_target,
                #gamma=args.gamma,
                learning_rate=args.lr_rate,
                seed=args.model_seed,
                tensorboard_log=args.log_path,
                policy_kwargs=args.policy_kwargs

                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1()])

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
    lr_rate = 1e-5
    # Permitir mayor exploración con un rango de clipping más amplio
    clip_range = 0.2
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.05
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[64, 64]
    )
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    kl_target = 0.015
    gae_lambda = 0.95
    vf_coef = 0.5


def run_experiment_params1(want_to_train=True):
    start_time = datetime.now()
    args = Params1()
    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = CornerEnv(args.model_total_timesteps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps,
                verbose=args.model_verbose,
                target_kl=args.kl_target,
                gamma=args.gamma,
                learning_rate=args.lr_rate,
                seed=args.model_seed,
                tensorboard_log=args.log_path
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([StopExperimentCallback(), HParamCallback1()])

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


def run_experiment_defaults(want_to_train=True):
    start_time = datetime.now()
    args = DefaultParams()
    # Initialize the environment
    env = CornerEnv(args.model_total_timesteps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path=args.log_path, name_prefix='rl_model')

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
        custom_callback_list = CallbackList([StopExperimentCallback(), HParamCallback1(), checkpoint_callback])

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

class Params3:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./logs_2024-11_04_CornerEnvShaped20_params3/"
    save_model_path = "./logs_2024-11_04_CornerEnvShaped20_params3/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "2024-11-04"
    # Considerar más recompensas futuras
    gamma = 0.99
    #  tasa de aprendizaje
    lr_rate = 2e-4
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.9
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.05


def run_experiment_CornerEnvShaped20_v3(want_to_train=True):
    start_time = datetime.now()
    args = Params3()
    if want_to_train is False:
        args.log_path += "test/"
    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = CornerEnvShaped20(args.model_total_timesteps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                target_kl=args.kl_target,
                gamma=args.gamma,
                learning_rate=args.lr_rate,
                seed=args.model_seed,
                gae_lambda=args.gae_lambda,
                tensorboard_log=args.log_path,
                policy_kwargs=args.policy_kwargs,
                vf_coef=args.vf_coef
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1()])

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


class Params4:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./logs_2024-11_05/"
    save_model_path = "./logs_2024-11_05/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "2024-11-05"
    # Considerar más recompensas futuras
    gamma = 0.99
    #  tasa de aprendizaje
    lr_rate = 2e-4
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.9
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.05


def run_experiment_CornerEnvShapedReward(want_to_train=True):
    start_time = datetime.now()
    args = Params4()
    if want_to_train is False:
        args.log_path += "test/"
    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = CornerEnvShapedReward(args.model_total_timesteps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                target_kl=args.kl_target,
                gamma=args.gamma,
                learning_rate=args.lr_rate,
                seed=args.model_seed,
                gae_lambda=args.gae_lambda,
                tensorboard_log=args.log_path,
                policy_kwargs=args.policy_kwargs,
                vf_coef=args.vf_coef
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])

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

class Params5:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241201_SimpleCorner_default/"
    save_model_path = "./20241201_SimpleCorner_default/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "20241201_SimpleCorner_default"
    # Considerar más recompensas futuras
    gamma = 0.99
    #  tasa de aprendizaje
    lr_rate = 2e-4
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.95
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.01

def run_experiment_SimpleCornerEnv_w(want_to_train=True, det=True):
    start_time = datetime.now()
    args = Params5()

    if want_to_train is False:
        if det is True:
            args.log_path += "test_deterministic_True/"
        else:
            args.log_path += "test_deterministic_False/"

    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = SimpleCornerEnv_w(args.max_episode_steps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                #target_kl=args.kl_target,
                #gamma=args.gamma,
                #learning_rate=args.lr_rate,
                seed=args.model_seed,
                #gae_lambda=args.gae_lambda,
                tensorboard_log=args.log_path
                #policy_kwargs=args.policy_kwargs,
                #vf_coef=args.vf_coef
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])

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
                                                  deterministic=det,
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

class Params6:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 60_000
    model_verbose = True
    log_path = "./20241201_SimpleCorner_myreward/"
    save_model_path = "./20241201_SimpleCorner_myreward/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "20241201_SimpleCorner_myreward"
    # Considerar más recompensas futuras
    gamma = 0.99
    #  tasa de aprendizaje
    lr_rate = 2e-4
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.95
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.01

def run_experiment_SimpleCornerEnv(want_to_train=True, det=True):
    start_time = datetime.now()
    args = Params6()

    if want_to_train is False:
        if det is True:
            args.log_path += "test_deterministic_True/"
        else:
            args.log_path += "test_deterministic_False/"

    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = SimpleCornerEnv(args.max_episode_steps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                # target_kl=args.kl_target,
                #gamma=args.gamma,
                #learning_rate=args.lr_rate,
                seed=args.model_seed,
                #gae_lambda=args.gae_lambda,
                tensorboard_log=args.log_path
                #policy_kwargs=args.policy_kwargs,
                #vf_coef=args.vf_coef
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])

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
                                                  deterministic=det,
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


class Params7:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241202_shaped20/"
    save_model_path = "./20241202_shaped20/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "20241202_shaped20"
    # Considerar más recompensas futuras
    gamma = 0.99
    #  tasa de aprendizaje
    lr_rate = 2e-4
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.95
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.01

def run_experiment_SimpleCornerEnvShaped20(want_to_train=True, det=True):
    start_time = datetime.now()
    args = Params7()

    if want_to_train is False:
        if det is True:
            args.log_path += "test_deterministic_True/"
        else:
            args.log_path += "test_deterministic_False/"

    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = SimpleCornerEnvShaped20(args.max_episode_steps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                # target_kl=args.kl_target,
                #gamma=args.gamma,
                #learning_rate=args.lr_rate,
                seed=args.model_seed,
                #gae_lambda=args.gae_lambda,
                tensorboard_log=args.log_path
                #policy_kwargs=args.policy_kwargs,
                #vf_coef=args.vf_coef
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])

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
                                                  deterministic=det,
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

class Params8:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    max_episode_steps=10_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241206/config1/"
    save_model_path = "./20241206/config1/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "./20241206-config1"
    # Considerar más recompensas futuras
    gamma = 0.99
    #  tasa de aprendizaje
    lr_rate = 1e-4
    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.95
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.01
    batch_size = 128
    n_steps = 4096
    n_epochs = 5

def run_experiment_SimpleCornerEnvRF(want_to_train=True, det=True):
    start_time = datetime.now()
    args = Params8()

    if want_to_train is False:
        if det is True:
            args.log_path += "test_deterministic_True/"
        else:
            args.log_path += "test_deterministic_False/"

    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = SimpleCornerEnvRF(args.max_episode_steps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                learning_rate=args.lr_rate,
                seed=args.model_seed,
                tensorboard_log=args.log_path,
                ent_coef=args.ent_coef,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs
                #policy_kwargs=args.policy_kwargs,
                #vf_coef=args.vf_coef,
                # target_kl=args.kl_target,
                # gamma=args.gamma,
                # gae_lambda=args.gae_lambda,
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])

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
                                                  deterministic=det,
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


class Params10:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    max_episode_steps=50_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241206_shaped20/config1/"
    save_model_path = "./20241206_shaped20/config1/ppo_model_pilot_room1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    tb_log_name = "20241206_shaped20"
    # Considerar más recompensas futuras
    gamma = 0.99

    # Cambia la arquitectura de la red a dos capas
    policy_kwargs = dict(
        net_arch=[128, 128]
    )
    kl_target = 0.03
    gae_lambda = 0.95
    vf_coef = 0.5
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef = 0.01
    batch_size = 128
    n_steps = 4096
    n_epochs = 5
    #  tasa de aprendizaje
    lr_rate = 1e-4

def run_experiment_SimpleCornerEnvShaped20_v2(want_to_train=True, det=True):
    start_time = datetime.now()
    args = Params10()

    if want_to_train is False:
        if det is True:
            args.log_path += "test_deterministic_True/"
        else:
            args.log_path += "test_deterministic_False/"

    # Crear el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Initialize the environment
    env = SimpleCornerEnvShaped20(args.max_episode_steps)
    env.set_trajectory_path(args.log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=args.log_path, name_prefix='rl_model')

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                verbose=args.model_verbose,
                learning_rate=args.lr_rate,
                seed=args.model_seed,
                tensorboard_log=args.log_path,
                ent_coef=args.ent_coef,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs
                # target_kl=args.kl_target,
                #gamma=args.gamma,
                #learning_rate=args.lr_rate,
                #gae_lambda=args.gae_lambda,
                #policy_kwargs=args.policy_kwargs,
                #vf_coef=args.vf_coef
                )
    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE ")
        custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])

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
                                                  deterministic=det,
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

def init_env_log_path(env,log_path="./logs"):
    env.set_trajectory_path(log_path)
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=log_path, name_prefix='rl_model')
    env = Monitor(env, filename=log_path,
                  info_keywords=env.get_info_keywords())
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard", "log"])
    custom_callback_list = CallbackList([HParamCallback1(),StopExperimentCallback(),checkpoint_callback])
    return new_logger,custom_callback_list

class Params11:

    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241206_sb3algos_v2/"
    tb_log_name = "20241206"

def run_experiment_SimpleCornerEnvShaped20RLAlgos(sb3_alg="PPO"):
    start_time = datetime.now()
    args = Params11()
    model = None
    # Initialize the environment
    env = SimpleCornerEnvShaped20(args.max_episode_steps)

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()
    log_dir = args.log_path
    match sb3_alg:
        case "DQN":
            log_dir += "DQN/"
            print(log_dir)
            model = DQN('MlpPolicy', env, verbose=2,
                        seed=args.model_seed, tensorboard_log=log_dir )
        case "A2C":
            log_dir += "A2C/"
            print(log_dir)
            model = A2C('MlpPolicy', env, verbose=True,
                        seed=args.model_seed, tensorboard_log=log_dir)
        case "PPO":
            log_dir += "PPO/"
            print(log_dir)
            model = PPO('MlpPolicy', env, verbose=True,
                        seed=args.model_seed, tensorboard_log=log_dir )
        # Crear el directorio si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    new_logger, custom_callback_list = init_env_log_path(env, log_dir)
    model.set_logger(new_logger)

    # start training
    model.learn(total_timesteps=args.model_total_timesteps,
                callback=custom_callback_list,
                tb_log_name=args.tb_log_name,
                log_interval=1)
    # Save the learned model
    model.save(log_dir+"final_model")

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")
    end_time = datetime.now()
    # Calcular la duración
    duration = end_time - start_time
    save_experiment_time(start_time, end_time, log_dir)

class Params12:

    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241208_Corner1RF_sb3algos/"
    tb_log_name = "20241208"

def run_experiment_SimpleCornerEnvRFAlgos(sb3_alg="PPO"):
    start_time = datetime.now()
    args = Params12()
    model = None
    # Initialize the environment
    env = SimpleCornerEnvRF(args.max_episode_steps)

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()
    log_dir = args.log_path
    match sb3_alg:
        case "DQN":
            log_dir += "DQN/"
            print(log_dir)
            model = DQN('MlpPolicy', env, verbose=2,
                        seed=args.model_seed, tensorboard_log=log_dir )
        case "A2C":
            log_dir += "A2C/"
            print(log_dir)
            model = A2C('MlpPolicy', env, verbose=True,
                        seed=args.model_seed, tensorboard_log=log_dir)
        case "PPO":
            log_dir += "PPO/"
            print(log_dir)
            model = PPO('MlpPolicy', env, verbose=True,
                        seed=args.model_seed, tensorboard_log=log_dir )
        # Crear el directorio si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    new_logger, custom_callback_list = init_env_log_path(env, log_dir)
    model.set_logger(new_logger)

    # start training
    model.learn(total_timesteps=args.model_total_timesteps,
                callback=custom_callback_list,
                tb_log_name=args.tb_log_name,
                log_interval=1)
    # Save the learned model
    model.save(log_dir+"final_model")

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")
    end_time = datetime.now()
    # Calcular la duración
    duration = end_time - start_time
    save_experiment_time(start_time, end_time, log_dir)

class Params13:

    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241208_RS10_sb3algos/"
    tb_log_name = "20241208"

def run_experiment_SimpleCornerEnvRS10(sb3_alg="PPO"):
    start_time = datetime.now()
    args = Params13()
    model = None
    # Initialize the environment
    env = SimpleCornerEnvRS10(args.max_episode_steps)

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()
    log_dir = args.log_path
    match sb3_alg:
        case "DQN":
            log_dir += "DQN/"
            print(log_dir)
            model = DQN('MlpPolicy', env, verbose=2,
                        seed=args.model_seed, tensorboard_log=log_dir )
        case "A2C":
            log_dir += "A2C/"
            print(log_dir)
            model = A2C('MlpPolicy', env, verbose=True,
                        seed=args.model_seed, tensorboard_log=log_dir)
        case "PPO":
            log_dir += "PPO/"
            print(log_dir)
            model = PPO('MlpPolicy', env, verbose=True,
                        seed=args.model_seed, tensorboard_log=log_dir )
        # Crear el directorio si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    new_logger, custom_callback_list = init_env_log_path(env, log_dir)
    model.set_logger(new_logger)

    # start training
    model.learn(total_timesteps=args.model_total_timesteps,
                callback=custom_callback_list,
                tb_log_name=args.tb_log_name,
                log_interval=1)
    # Save the learned model
    model.save(log_dir+"final_model")

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")
    end_time = datetime.now()
    # Calcular la duración
    duration = end_time - start_time
    save_experiment_time(start_time, end_time, log_dir)

if __name__ == '__main__':
    #run_experiment_default_noisy_obs(want_to_train=False)
    #run_experiment_params1(want_to_train=True)
    # run_experiment_CornerEnvShapedReward(want_to_train=True)
    #run_experiment_CornerEnvShapedReward(want_to_train=False)
    # run_experiment_SimpleCornerEnv(want_to_train=True)
    #run_experiment_SimpleCornerEnv(want_to_train=False, det= True)
    # run_experiment_SimpleCornerEnvShaped20_v2(want_to_train=False, det=False)
    # run_experiment_SimpleCornerEnvRF(want_to_train=True, det=False)
    # lo siguiente da error
    #run_experiment_SimpleCornerEnvShaped20RLAlgos("DQN")
    #run_experiment_SimpleCornerEnvShaped20RLAlgos("A2C")
    #run_experiment_SimpleCornerEnvShaped20RLAlgos("PPO")
    # run_experiment_SimpleCornerEnvRFAlgos("DQN")
    # run_experiment_SimpleCornerEnvRFAlgos("A2C")
    # run_experiment_SimpleCornerEnvRFAlgos("PPO")
    # ULTIMOS UTILIZADOS:
    # run_experiment_SimpleCornerEnvRS10("DQN")
    # run_experiment_SimpleCornerEnvRS10("A2C")
    run_experiment_SimpleCornerEnvRS10("PPO")




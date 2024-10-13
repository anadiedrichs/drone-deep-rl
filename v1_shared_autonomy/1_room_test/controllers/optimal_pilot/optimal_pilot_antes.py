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
from gymnasium import ObservationWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

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

class DefaultParams:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 7
    model_total_timesteps = 40_000
    model_verbose = True
    log_path = "./logs-ent-2024-10-12/"
    save_model_path = "./logs-ent-2024-10-12/ppo_model_pilot_room_1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
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

# Definir el wrapper personalizado para añadir ruido aditivo a las observaciones
class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std  # Desviación estándar del ruido

    def observation(self, obs):
        # Sumar ruido gaussiano a las observaciones
        noisy_obs = obs + np.random.normal(0, self.noise_std, size=obs.shape)
        # Limitar las observaciones al rango [-1, 1] (si es necesario)
        noisy_obs = np.clip(noisy_obs, -1, 1)
        return noisy_obs
def run_experiment_default_noisy_obs(want_to_train=True):
    start_time = datetime.now()
    args = DefaultParams()
    # Initialize the environment
    env = CornerEnv()

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()
    # Aplicar el wrapper de ruido aditivo
    env = NoisyObservationWrapper(env, noise_std=0.05)

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
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
    save_experiment_time(start_time,end_time,args.log_path)

class Params(DefaultParams):
    """
    Common model hyperparameters for PPO and
    other configs
    """
    kl_target = 0.015
    gae_gamma = 0.9
    gae_lambda = 0.95
    batch_size = 128
    n_steps = 2048
    lr_rate = 0.0001
    # Cambia la arquitectura de la red a dos capas de 32 neuronas cada una
    policy_kwargs = dict(
        net_arch=[32, 32]
    )
    ls_rate = 0.001  # linear schedule rate
    ent_coef=0.07

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
    clip_range=0.2
    # Mayor coeficiente de entropía para fomentar la exploración
    ent_coef=0.05
    # Cambia la arquitectura de la red a dos capas de 32 neuronas cada una
    policy_kwargs = dict(
        net_arch=[16, 16]
    )
    # NO ESTOY USANDO LOS SIGUIENTES
    ls_rate = 0.001  # linear schedule rate
    kl_target = 0.015
    gae_lambda = 0.95


def run_experiment_ent_coef(want_to_train=True):
    start_time = datetime.now()
    args = Params1()
    # Initialize the environment
    env = CornerEnv()

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()
    # Aplicar el wrapper de ruido aditivo
    #env = NoisyObservationWrapper(env, noise_std=0.05)

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
    save_experiment_time(start_time,end_time,args.log_path)



def run_experiment(want_to_train=True):
    start_time = datetime.now()
    args = DefaultParams()
    # Initialize the environment
    env = CornerEnv()

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()
    # Aplicar el wrapper de ruido aditivo
    env = NoisyObservationWrapper(env, noise_std=0.05)

    env = Monitor(env, filename=args.log_path,
                  info_keywords=env.get_info_keywords())

    # model to train: PPO
    model = PPO('MlpPolicy', env,
                n_steps=args.n_steps,
                verbose=args.model_verbose,
                target_kl=args.kl_target,
                batch_size=args.batch_size,
                gae_lambda=args.gae_lambda,
                learning_rate=linear_schedule(args.ls_rate),
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

class Params2(DefaultParams):
    """
    Common model hyperparameters for PPO and
    other configs
    """
    kl_target = 0.015
    gae_gamma = 0.9
    gae_lambda = 0.95
    batch_size = 128
    n_steps = 2048
    lr_rate = 0.0001
    # Cambia la arquitectura de la red a dos capas de 32 neuronas cada una
    policy_kwargs = dict(
        net_arch=[32, 32]
    )
    ls_rate = 0.001  # linear schedule rate
    ent_coef=0.07
# tensorboard hyperparameter logging
class HDQNParamCallback(BaseCallback):

    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "buffer_size" : self.model.buffer_size,  # Tamaño del buffer de experiencia
            "learning_starts" : self.model.learning_starts,  # Pasos antes de empezar a aprender
            "target_update_interval" : self.model.target_update_interval,  # Intervalo para actualizar la red objetivo
            "exploration_fraction" : self.model.exploration_fraction,  # Fracción del entrenamiento para explorar
            "exploration_final_eps" : self.model.exploration_final_eps,  # Epsilon final para epsilon-greedy
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "seed": self.model.seed,
            "total_timesteps":self.model._total_timesteps
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
            hparam_dict,
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

def run_experiment_dqn(want_to_train=True):
    start_time = datetime.now()
    LOG_DIR = "./logs-dqn_2/"
    # Initialize the environment
    env = CornerEnv()

    # Aplicar el monitor para registrar los episodios
    env = Monitor(env, filename=LOG_DIR,
                  info_keywords=env.get_info_keywords())

    # Configurar el modelo DQN
    model = DQN('MlpPolicy', env,
                verbose=1,
                batch_size=64,           # Tamaño de batch
                learning_rate=1e-4,       # Tasa de aprendizaje ajustable
                buffer_size=1000,        # Tamaño del buffer de experiencia
                learning_starts=1000,     # Pasos antes de empezar a aprender
                target_update_interval=500,  # Intervalo para actualizar la red objetivo
                gamma=0.98,               # Descuento de recompensas
                exploration_fraction=0.7,  # Fracción del entrenamiento para explorar
                exploration_final_eps=0.2, # Epsilon final para epsilon-greedy
                tensorboard_log=LOG_DIR,
                policy_kwargs=dict(net_arch=[16, 16]),  # Ejemplo con una red más pequeña
                )
    # set up logger
    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    if want_to_train:
        print("TRAINING MODE")
        custom_callback_list = CallbackList([StopExperimentCallback(),HDQNParamCallback()])

        model.learn(total_timesteps=40_000,callback=custom_callback_list)
        model.save(LOG_DIR+str("dqn_model_file"))

    else:
        print("TESTING MODE")
        model.load(LOG_DIR+str("dqn_model_file"))
        obs = env.reset()
        # Evaluate the agent policy
        mean_reward, std_reward = evaluate_policy(model, env,
                                                  n_eval_episodes=10,
                                                  deterministic=True,
                                                  return_episode_rewards=True)
        data = {'Reward': mean_reward, 'Len': std_reward}
        # Create DataFrame
        df = pd.DataFrame(data)
        print("Results")
        print(df)
        # Save DataFrame to CSV
        df.to_csv(LOG_DIR+"/evaluation_results_dqn.csv", index=False)

    # Cerrar el simulador
    # env.simulationQuit(0)
    print("run_experiment_dqn ended")
    end_time = datetime.now()
    # Calcular la duración
    save_experiment_time(start_time, end_time, LOG_DIR)


if __name__ == '__main__':

    # run_experiment_defaults(want_to_train=False)
    #run_experiment_default_noisy_obs(want_to_train=False)
    #run_experiment_ent_coef(want_to_train=True)
    run_experiment_dqn(want_to_train=True)


"""
Copilot experiment for room 1
"""

import sys
import os
import torch
import torch.nn.functional as F
from typing import Callable
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from bc_expert.bc_expert import *
from trajectory_logger.DroneTrajectoryLogger import DroneTrajectoryLogger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from pilots.pilot import *
from pilots.LaggyPilot import *
from pilots.NoisyPilot import *
from copilot.Copilots import CopilotCornerEnv

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



class GuidanceCallback(BaseCallback):
    def __init__(self, bc_policy, lambda_reg=0.1, verbose=0):
        super().__init__(verbose)
        self.bc_policy = bc_policy  # Política entrenada con imitation learning
        self.lambda_reg = lambda_reg  # Peso de regularización

    def _on_step(self) -> bool:
        try:
            # Obtener observaciones y corregir su forma
            obs = self.locals['rollout_buffer'].observations
            print(f"Forma de las observaciones durante el entrenamiento: {obs.shape}")

            # Asegurar que las observaciones tengan forma correcta
            obs = obs.reshape(-1, 11)
            print(f"Forma de las observaciones corregida: {obs.shape}")

            # Usar solo las primeras 10 entradas para la política de imitation learning
            obs_10 = obs[:, :-1]
            print(f"Forma de obs_10 para bc_policy: {obs_10.shape}")

            # Predecir acciones con la política PPO
            ppo_actions, _ = self.model.policy.predict(obs, deterministic=False)
            ppo_actions = torch.tensor(ppo_actions, dtype=torch.float32)
            print(f"Tipo de ppo_actions: {type(ppo_actions)}, forma: {ppo_actions.shape}")

            # Predecir acciones óptimas con la política entrenada
            optimal_actions = []
            for obs_single in obs_10:
                action, _ = self.bc_policy.predict(obs_single)
                # print(f"action tipo: {type(action)}, forma: {optimal_actions.shape}")
                # Convertir la acción predicha en un array para compatibilidad
                action = np.asarray(action, dtype=np.float32)
                optimal_actions.append(action)

            # Verificar que las acciones sean numéricas y tengan forma consistente
            optimal_actions = np.array(optimal_actions, dtype=np.float32)
            print("optimal_actions.shape "+str(len(optimal_actions.shape)))
            #     optimal_actions = optimal_actions[:, np.newaxis]  # Asegurar dimensión correcta

            # Convertir a tensor de PyTorch
            optimal_actions = torch.tensor(optimal_actions, dtype=torch.float32)
            print(f"Tipo de optimal_actions: {type(optimal_actions)}, forma: {optimal_actions.shape}")

            # Calcular divergencia KL
            kl_div = F.kl_div(torch.log(ppo_actions), optimal_actions, reduction="batchmean")

            # Ajustar las recompensas con la regularización
            self.locals['rollout_buffer'].rewards += self.lambda_reg * kl_div.item()

        except Exception as e:
            print(f"Error durante el paso del callback: {e}")
            return False  # Detener el entrenamiento si hay un error crítico

        return True


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    alpha = 0.5
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
    pilot_model_path = "../bc_expert/imitation/logs/bc_policy"
    save_model_path = "./logs/ppo_model"  # os.path.join(log_path,"ppo_model_pilot_room_1")
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    copilot_alpha = 0.5  # copilot level of tolerance

def init_copilot_policy(model,pilot_model):

    # Transferir los pesos de las primeras 10 entradas de `pilot_model` a `model`
    with torch.no_grad():
        # Copiar pesos de las 10 primeras entradas
        model.policy.mlp_extractor.policy_net[0].weight[:, :10] = pilot_model.policy.mlp_extractor.policy_net[0].weight

        # Inicializar aleatoriamente los pesos de la entrada adicional
        model.policy.mlp_extractor.policy_net[0].weight[:, 10] = torch.randn_like(
            model.policy.mlp_extractor.policy_net[0].weight[:, 10]
        )

        # Copiar sesgos (bias) de la política de `pilot_model`
        model.policy.mlp_extractor.policy_net[0].bias = pilot_model.policy.mlp_extractor.policy_net[0].bias
        # check the initialization
        print(model.policy.mlp_extractor.policy_net[0].weight)
        print(model.policy.mlp_extractor.policy_net[0].bias)


def run_experiment(seed: float, alpha: float, is_pilot_noisy: bool, copilot_alpha: float, want_to_train: False):

    args = Params()

    if alpha is not None:
        args.alpha = alpha

    if copilot_alpha is not None:
        args.copilot_alpha = alpha

    if seed is not None:
        args.model_seed = seed

    # args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    # pilot_model = PPO.load(args.pilot_model_path)

    # pilot_model is the bc_policy,the model policy trained with imitation learning
    # pilot optimal model
    pilot_model = load_create_bc_model(args.pilot_model_path, args.log_path)

    if is_pilot_noisy:
        args.log_path = "./logs-noisy/"
        args.save_model_path = os.path.join(args.log_path, "model_noisy_copilot_room_1")
        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)
    else:
        args.log_path = "logs-laggy/"
        args.save_model_path = os.path.join(args.log_path, "model_laggy_copilot_room_1")
        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)

    # Initialize the environment
    env = CopilotCornerEnv( model=pilot_model, pilot = pilot, seed=args.model_seed, alpha=args.copilot_alpha,)
    env.set_trajectory_path(args.log_path)
    env.training_mode = want_to_train
    print("want_to_train : " + str(want_to_train))
    env = Monitor(env, filename=args.eval_result_file, info_keywords=("is_success",))

    # Configuración de PPO con capas ocultas de 32 neuronas
    policy_kwargs = dict(
        net_arch=dict(
            pi=[32, 32],  # Capas ocultas para la red de política (32 neuronas cada una)
            vf=[32, 32]  # Capas ocultas para la red de valor (32 neuronas cada una)
        )
    )

    # Crear el modelo PPO con las capas ajustadas
    model = PPO(
        "MlpPolicy",  # Política de perceptrón multicapa
        env,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    init_copilot_policy(model, pilot_model)

    # set up logger
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)

    env = DummyVecEnv([lambda: env])

    # Accede al entorno interno y reinícialo con la semilla
    if hasattr(env, 'envs') and len(env.envs) > 0:
        obs = env.envs[0].reset(seed=args.model_seed)
    else:
        obs = env.reset()
    # Extraer solo observaciones si env.reset() retorna un tuple
    if isinstance(obs, tuple):
        obs, _ = obs

    print(f"Forma de las observaciones al reiniciar: {obs.shape}")

    # Evaluate the policy
    if want_to_train:
        # custom_callback = StopExperimentCallback()
        custom_callback = GuidanceCallback(pilot_model.policy, lambda_reg=0.1)
        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback)
        # Save the learned model
        model.save(args.save_model_path)

    else:
        # evaluate the agent
        # Load a saved model
        print(args.save_model_path)
        model.load(args.save_model_path)
        # obs = env.reset()
        env.set_model(model)
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


def train_copilot(seed: float, alpha: float, is_pilot_noisy: bool):
    args = Params()

    # args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    pilot_model = PPO.load(args.pilot_model_path)

    if alpha is not None:
        args.copilot_alpha = alpha
    if seed is not None:
        args.model_seed = seed



    run_experiment(pilot, True, args)


def test_copilot(seed: float, alpha: float, is_pilot_noisy: bool, copilot_alpha: float):
    args = Params()

    # args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    pilot_model = PPO.load(args.pilot_model_path)

    if alpha is not None:
        args.alpha = alpha
    else:
        alpha = 0.5
    if copilot_alpha is not None:
        args.copilot_alpha = alpha
    else:
        copilot_alpha = 0.5
    if seed is not None:
        args.model_seed = seed

    if is_pilot_noisy:
        # where is the model to load
        args.save_model_path = "./logs-noisy/model_noisy_copilot_room_1"
        args.log_path = "./logs-noisy_test/"
        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(pilot_model, args.model_seed)
    else:
        args.save_model_path = "logs-laggy_1/model_laggy_copilot_room_1"
        args.log_path = "./logs-laggy_test/"
        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(pilot_model, args.model_seed)

    run_experiment(pilot,False, args)


def run_experiment_train_copilot_for_laggy_pilot():

    log_path = "./laggy_pilot_training_logs/"
    # path to expert / optimal pilot model 
    model_path = "../bc_expert/imitation/logs/bc_policy"
    pilot_model = load_create_bc_model(model_path, log_path)

    # Configuración del logger para entrenamiento
    logger = DroneTrajectoryLogger(
        use_copilot=True,  # Activar copiloto (en modo observador / entrenar)
        pilot_type="Laggy",  # Usar LaggyPilot
        model=pilot_model,
        model_seed=42,
        model_total_timesteps=10000,
        log_path=log_path,
        n_eval_episodes=10,
        pilot_alpha=0.5,  # Configuración para el LaggyPilot
        copilot_alpha=0.0  # Copilot en modo observador (sin tomar decisiones)
    )

    # Ejecutar el experimento de recolección de datos
    logger.run_experiment(file_name="laggy_0_5",debug=True, train = True)


if __name__ == '__main__':
    # train laggy copilot
    # run_experiment(is_pilot_noisy=True, seed=5, alpha=0.5, copilot_alpha=0.5, want_to_train= True)
    run_experiment_train_copilot_for_laggy_pilot()
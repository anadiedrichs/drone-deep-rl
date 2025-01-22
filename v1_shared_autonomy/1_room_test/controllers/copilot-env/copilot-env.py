"""
Copilot experiment for room 1
"""

# 1) IMPORTS

# Standard imports
import sys
import os

# External libraries
import torch
import torch.nn.functional as F
from typing import Callable
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.logger import HParam

# Local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from bc_expert.bc_expert import *
from trajectory_logger.DroneTrajectoryLogger import DroneTrajectoryLogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from pilots.pilot import *
from pilots.LaggyPilot import *
from pilots.NoisyPilot import *
from copilot.Copilots import CopilotCornerEnv
# 1.1 logging level for lib sb4
import logging
import stable_baselines3

stable_baselines3.common.logger.LOGGING_LEVEL = logging.DEBUG


# 2) PARAMETERS AND CONFIGURATIONS
class Params:
    """
    Contains common hyperparameters and configurations for the experiment.
    """
    alpha = 0.5
    model_seed = 5
    model_total_timesteps = 10_000
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


# 3) CALLBACKS
class StopExperimentCallback(BaseCallback):
    """
    Stops the training process based on specific termination conditions.
    """

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
        if cumm_score <= -10_000 or cumm_score > 10_000:
            self.model.env.envs[0].truncated = True
            self.model.env.envs[0].is_success = False
            self.logger.record("is_success", 0)
            return False  # stop the training

        return True  # # Continue training


class HParamCallback1(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training,
     and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate(1) if callable(
                self.model.learning_rate) else self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "seed": self.model.seed,
            "target_kl": self.model.target_kl,
            "ent_coef": self.model.ent_coef,
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
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class BetaAdjuster:
    """
    Clase para ajustar el valor de beta dinámicamente durante el entrenamiento.
    """

    def __init__(self, initial_beta=0.0, max_beta=1.0, total_episodes=500):
        self.beta = initial_beta  # Valor inicial de beta
        self.max_beta = max_beta  # Valor máximo de beta
        self.total_episodes = total_episodes  # Total de episodios para incrementar beta

    def update(self, current_episode: int):
        """
        Incrementa beta progresivamente basado en el número de episodios.
        """
        self.beta = min(self.max_beta, current_episode / self.total_episodes)
        return self.beta

    def update_using_reward(self, avg_reward: float):
        self.beta = max(0.0, min(1.0, (avg_reward + 1) / 2))
        return self.beta


class GuidanceCallback(BaseCallback):
    """
     Callback para guiar el entrenamiento del copiloto mediante:
    1. Penalización de recompensas basada en la divergencia KL entre la política PPO y la política BC.
    2. Ajuste dinámico de beta basado en las recompensas acumuladas promedio.

    """

    def __init__(self, bc_policy, beta_adjuster, lambda_reg=0.01, reward_window_size=100, verbose=1):
        super().__init__(verbose)
        self.bc_policy = bc_policy  # Pre-trained BC policy
        self.lambda_reg = lambda_reg  # Regularization weight
        self.beta_adjuster = beta_adjuster  # Objeto que gestiona beta
        self.reward_window_size = reward_window_size  # Tamaño de la ventana para promediar recompensas
        self.episode_rewards = []  # Recompensas acumuladas recientes
        self.episode_count = 0  # Contador de episodios

    def _increment_episode_count(self):
        """
        Incrementa el contador de episodios si algún episodio ha terminado.
        """
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            if any(dones):  # Si algún episodio terminó
                self.episode_count += sum(dones)  # Incrementa el contador
                if self.verbose > 0:
                    print(f"Episodios completados: {self.episode_count}")

    def _on_step(self) -> bool:
        # Llama al método privado para incrementar el contador
        self._increment_episode_count()
        if self.episode_count >= self.beta_adjuster.total_episodes:
            print("Max number of episodes reached")
            return False
        try:
            print("Rollout buffer content:", self.locals['rollout_buffer'])
            # Get observations and adjust their shape
            obs = self.locals['rollout_buffer'].observations
            # print(f"Forma de las observaciones durante el entrenamiento: {obs.shape}")
            obs = obs.reshape(-1, 11)
            # print(f"Forma de las observaciones corregida: {obs.shape}")
            # Use the first 10 inputs for the BC policy
            obs_10 = obs[:, :-1]
            # print(f"Forma de obs_10 para bc_policy: {obs_10.shape}")
            # Predict actions with PPO
            ppo_actions, _ = self.model.policy.predict(obs, deterministic=False)
            ppo_actions = torch.tensor(ppo_actions, dtype=torch.float32)
            print(f"Tipo de ppo_actions: {type(ppo_actions)}, forma: {ppo_actions.shape}")
            print(ppo_actions[0])
            # Predecir acciones óptimas con la política entrenada
            # optimal_actions = []
            # for obs_single in obs_10:
            #     action, _ = self.bc_policy.predict(obs_single)
            #     # print(f"action tipo: {type(action)}, forma: {optimal_actions.shape}")
            #     # Convertir la acción predicha en un array para compatibilidad
            #    action = np.asarray(action, dtype=np.float32)
            #    optimal_actions.append(action)

            # Predict optimal actions using the BC policy
            optimal_actions = np.array([
                np.asarray(self.bc_policy.predict(obs_single)[0], dtype=np.float32)
                for obs_single in obs_10
            ])
            optimal_actions = torch.tensor(optimal_actions, dtype=torch.float32)

            # Verificar que las acciones sean numéricas y tengan forma consistente
            # print("optimal_actions.shape "+str(len(optimal_actions.shape)))
            #     optimal_actions = optimal_actions[:, np.newaxis]  # Asegurar dimensión correcta

            # Convertir a tensor de PyTorch
            # optimal_actions = torch.tensor(optimal_actions, dtype=torch.float32)
            print(f"Tipo de optimal_actions: {type(optimal_actions)}, forma: {optimal_actions.shape}")
            print(optimal_actions[0])

            #Convert discrete actions to one-hot-encoding
            num_actions = 6  # Número de acciones posibles (0 a 5) TODO hardcode
            # One-hot para ppo_actions
            ppo_actions_one_hot = F.one_hot(ppo_actions.long(), num_classes=num_actions).float()
            print(f"ppo_actions_one_hot: {ppo_actions_one_hot[0]}")
            # One-hot para optimal_actions
            optimal_actions_one_hot = F.one_hot(optimal_actions.long(), num_classes=num_actions).float()
            print(f"optimal_actions_one_hot: {optimal_actions_one_hot[0]}")
            print(f"Shape of ppo_actions_one_hot: {ppo_actions_one_hot.shape}")
            print(f"Shape of optimal_actions_one_hot: {optimal_actions_one_hot.shape}")
            print(f"first reward value : {self.locals['rollout_buffer'].rewards[0]}")
            # debe dar [2048, 6]
            # Calculate KL divergence
            # + 1e-10 es para evitar el log(cero)
            kl_div = F.kl_div(torch.log(ppo_actions_one_hot + 1e-10),
                              optimal_actions_one_hot,
                              reduction="batchmean")

            print(f"kl_div: {kl_div.item()}")
            # que el valor de kl me quede entre 0 y 5
            clamped_kl = torch.clamp(kl_div, min=0, max=5).item()
            # Media de las recompensas actuales
            rewards_tensor = torch.tensor(self.locals['rollout_buffer'].rewards, dtype=torch.float32)
            reward_scale = torch.mean(rewards_tensor).item()
            # scaled_kl = clamped_kl / (reward_scale + 1e-10)  # Evita divisiones por cero
            clamped_kl = torch.clamp(kl_div, min=0, max=5).item()
            scaled_kl = torch.clamp(
                torch.tensor(clamped_kl / (reward_scale + 1e-10), dtype=torch.float32),
                min=-0.5,
                max=0.5
            ).item()
            self.locals['rollout_buffer'].rewards -= self.lambda_reg * scaled_kl

            # Actualizar beta según recompensas acumuladas
            reward = rewards_tensor.mean().item()
            self.episode_rewards.append(reward)
            if len(self.episode_rewards) > self.reward_window_size:
                self.episode_rewards.pop(0)

            # Calcular recompensa promedio en la ventana
            # actualizar beta
            avg_reward = np.mean(self.episode_rewards)
            self.beta_adjuster.update_using_reward(avg_reward)
            #self.beta_adjuster.update(self.episode_count)
            if self.verbose > 0:
                print(f"KL divergence: {kl_div:.4f}, Scaled KL: {scaled_kl:.4f}, Beta: {self.beta_adjuster.beta:.4f}")
                print(f"Recompensa promedio: {avg_reward:.4f}")

        except Exception as e:
            print(f"Error during GuidanceCallback step: {e}")
            return False  # Stop training if an error occurs
        return True  # Continue training


def init_copilot_policy(model, pilot_model):
    """
    Transfers weights from the BC-trained policy to the PPO model's policy.
    """

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


def run_experiment(seed: int, alpha: float, is_pilot_noisy: bool, copilot_alpha: float, want_to_train: False):
    args = Params()

    if alpha is not None:
        args.alpha = alpha

    if copilot_alpha is not None:
        args.copilot_alpha = alpha

    if seed is not None:
        args.model_seed = seed

    # pilot_model is the bc_policy,the model policy trained with imitation learning
    # pilot optimal model
    pilot_model = load_create_bc_model(args.pilot_model_path, args.log_path)

    if is_pilot_noisy:
        if want_to_train:
            args.log_path = "logs-noisy-train/"
        else:
            args.log_path = "logs-noisy-test/"

        args.save_model_path = os.path.join(args.log_path, "model_noisy_copilot_room_1")
        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)
    else:
        if want_to_train:
            args.log_path = "logs-laggy-train/"
        else:
            args.log_path = "logs-laggy-test/"

        args.save_model_path = os.path.join(args.log_path, "model_laggy_copilot_room_1")
        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)

    # Inicializar BetaAdjuster
    beta_adjuster = BetaAdjuster(initial_beta=0.0, max_beta=1.0, total_episodes=500)
    # Initialize the environment
    env = CopilotCornerEnv(model=pilot_model,
                           pilot=pilot,
                           seed=args.model_seed,
                           alpha=args.copilot_alpha,
                           beta_adjuster=beta_adjuster)
    env.set_trajectory_path(args.log_path)
    env.training_mode = want_to_train
    print("want_to_train : " + str(want_to_train))
    env = Monitor(env, filename=args.eval_result_file, info_keywords=env.get_info_keywords())

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
        seed=args.model_seed,
        verbose=1,
        tensorboard_log=args.log_path)

    init_copilot_policy(model, pilot_model)

    # set up logger "stdout", "csv", "tensorboard", "log"
    new_logger = configure(args.log_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)
    obs, _ = env.reset(seed=args.model_seed)
    print(f"Forma de las observaciones al reiniciar: {obs.shape}")
    # train the copilot policy
    if want_to_train:
        # custom_callback = StopExperimentCallback()
        custom_callback_list = CallbackList([HParamCallback1(),
                                             GuidanceCallback(pilot_model.policy, beta_adjuster, lambda_reg=0.01, verbose=1)])
        # start training
        model.learn(total_timesteps=args.model_total_timesteps,
                    callback=custom_callback_list,
                    log_interval=1
                    )
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


def run_experiment_generate_trajectories_for_laggy_pilot():
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
    logger.run_experiment(file_name="laggy_0_5", debug=True, train=True)


if __name__ == '__main__':
    # generate a Trajectories dataset
    # run_experiment_generate_trajectories_for_laggy_pilot()
    # laggy train
    run_experiment(is_pilot_noisy=False, seed=5, alpha=0.3, copilot_alpha=0.5, want_to_train=True)
    # test laggy
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.3, copilot_alpha=0.5, want_to_train=False)
    # noisy train
    # run_experiment(is_pilot_noisy=True, seed=5, alpha=0.3, copilot_alpha=0.5, want_to_train= True)

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
    # parameters used to train the model
    alpha = 0.5
    model_seed = 5
    model_total_timesteps = 1_000_000
    model_total_episodes = 100
    model_verbose = True
    kl_target = 0.015
    #gae_gamma = 0.99
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 512
    ls_rate = 0.001  # linear schedule rate
    log_path = "./logs/" 
    pilot_model_path = "../bc_expert/imitation/logs_20250122_50/bc_policy"
    save_model_path = "./logs/ppo_model"  # os.path.join(log_path,"ppo_model_pilot_room_1")
    # parameters used to test / evaluate the model
    load_model_path = ""
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "results.csv")
    copilot_alpha = 0.5  # copilot level of tolerance
    eval_max_steps = 100_000


# 3) CALLBACKS


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
            #"gamma": self.model.gamma,
            #"gae_lambda": self.model.gae_lambda,
            #"n_steps": self.model.n_steps,
            #"batch_size": self.model.batch_size,
            "seed": self.model.seed,
            #"target_kl": self.model.target_kl,
            #"ent_coef": self.model.ent_coef,
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

    def __init__(self, initial_beta=0.0, max_beta=1.0, total_episodes=100):
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
        """
        Beta increases if the average reward is below 0.5, and decreases if it is above 0.5.
        """
        self.beta = max(0.0, min(1.0, 1-(avg_reward + 1) / 2))
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
        self._rewards_tensor = None
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

    def _update_rewards_to_guide_policy(self):

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
        # Media de las recompensas actuales
        reward_scale = torch.mean(self._rewards_tensor).item()
        # scaled_kl = clamped_kl / (reward_scale + 1e-10)  # Evita divisiones por cero
        # que el valor de kl me quede entre 0 y 5
        clamped_kl = torch.clamp(kl_div, min=0, max=5).item()
        scaled_kl = torch.clamp(
            torch.tensor(clamped_kl / (reward_scale + 1e-10), dtype=torch.float32),
            min=-0.5,
            max=0.5
        ).item()
        self.locals['rollout_buffer'].rewards -= self.lambda_reg * scaled_kl
        if self.verbose > 0:
            print(f"KL divergence: {kl_div: .4f}, Scaled KL: {scaled_kl: .4f}")

    def _on_step(self) -> bool:
        # Llama al método privado para incrementar el contador
        self._increment_episode_count()

        if self.episode_count >= self.beta_adjuster.total_episodes:
            print("Max number of episodes reached")
            return False

        try:
            self._rewards_tensor = torch.tensor(self.locals['rollout_buffer'].rewards, dtype=torch.float32)

            # self._update_rewards_to_guide_policy()

            # Actualizar beta según recompensas acumuladas
            reward = self._rewards_tensor.mean().item()
            self.episode_rewards.append(reward)
            if len(self.episode_rewards) > self.reward_window_size:
                self.episode_rewards.pop(0)

            # Calcular recompensa promedio en la ventana
            # actualizar beta
            avg_reward = np.mean(self.episode_rewards)
            self.beta_adjuster.update_using_reward(avg_reward)
            # self.beta_adjuster.update(self.episode_count)
            if self.verbose > 0:
                print(f"Beta: {self.beta_adjuster.beta:.4f}")
                print(f"Recompensa promedio: {avg_reward:.4f}")

        except Exception as e:
            print(f"Error during GuidanceCallback step: {e}")
            return False  # Stop training if an error occurs
        return True  # Continue training


def init_env_log_path(env, seed, pilot_model, log_path):
    env.set_trajectory_path(log_path)

    env = Monitor(env, filename=log_path,
                  info_keywords=env.get_info_keywords())
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard", "log"])
    # PPO Model initialization
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
        seed=seed,
        verbose=1,
        tensorboard_log=log_path)

    init_copilot_policy(model, pilot_model)

    model.set_logger(new_logger)

    return new_logger, model


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
        # print(model.policy.mlp_extractor.policy_net[0].weight)
        # print(model.policy.mlp_extractor.policy_net[0].bias)


def run_experiment(seed: int, alpha: float, is_pilot_noisy: bool, copilot_alpha: float, want_to_train: False):
    args = Params()

    if alpha is not None:
        args.alpha = alpha

    if copilot_alpha is not None:
        args.copilot_alpha = copilot_alpha

    if seed is not None:
        args.model_seed = seed

    # pilot_model is the bc_policy,the model policy trained with imitation learning
    # pilot optimal model
    pilot_model = load_create_bc_model(args.pilot_model_path, args.log_path)

    if is_pilot_noisy:
        if want_to_train:
            args.log_path = "logs-noisy-train/"
            args.save_model_path = os.path.join(args.log_path, "model_noisy_copilot_room_1")
        else:
            args.log_path = "logs-noisy-test/"
            args.load_model_path = "logs-noisy-train/model_noisy_copilot_room_1"

        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)
    else:
        if want_to_train:
            args.log_path = "logs-laggy-train/"
            args.save_model_path = os.path.join(args.log_path, "model_laggy_copilot_room_1")
        else:
            args.log_path = "logs-laggy-test/"
            args.load_model_path = "logs-laggy-train/model_laggy_copilot_room_1"

        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)

    # Inicializar BetaAdjuster
    beta_adjuster = BetaAdjuster(initial_beta=0.0, max_beta=1.0, total_episodes=args.model_total_episodes)
    # Initialize the environment
    env = CopilotCornerEnv(model=pilot_model,
                           pilot=pilot,
                           seed=args.model_seed,
                           alpha=args.copilot_alpha,
                           beta_adjuster=beta_adjuster)
    env.training_mode = want_to_train
    print("want_to_train : " + str(want_to_train))
    new_logger, model = init_env_log_path(env, args.model_seed, pilot_model, args.log_path)

    obs, _ = env.reset(seed=args.model_seed)
    # print(f"Forma de las observaciones al reiniciar: {obs.shape}")
    # train the copilot policy
    if want_to_train:
        # create callbacks
        env.set_model(pilot_model)
        checkpoint_callback = CheckpointCallback(save_freq=1_000, save_path=args.log_path + "models_bkp/")
        guide_callback = GuidanceCallback(pilot_model.policy, beta_adjuster)
        #custom_callback_list = CallbackList([HParamCallback1(), StopExperimentCallback(), checkpoint_callback])
        custom_callback_list = CallbackList([HParamCallback1(),checkpoint_callback, guide_callback])
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
        print(args.load_model_path)
        model.load(args.load_model_path)
        # obs = env.reset()
        env.set_model(model)
        # Evaluate the policy
        evaluate_copilot(env,model, args)

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")

def save_results(r,l,g,file_name):
    data = {'Reward': r, 'Len': l, 'Goal':g}
    # Create DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False,mode='w')

def evaluate_copilot(env,model, args:Params):
    # Initialize the environment
    #env = CornerEnv()
    obs, _ = env.reset(seed=args.model_seed)
    
    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    success = []
    steps = 0

    # Evaluate the policy
    while True:
        # print("obs=", obs)
        action, _ = model.predict(obs,deterministic=True)
        # Convertir la acción a un array
        # if isinstance(action, (int, float)):
        #    action = np.array([action])
        # Validar que la acción sea compatible con el entorno
        #if len(action.shape) == 0:
        #    action = np.expand_dims(action, axis=0)
        print("Action:", action, "Action Shape:", action.shape)
        print("Observation:", obs, "Observation Shape:", obs.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Para mantener la lógica original
        print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward
        if isinstance(info, list):
            info = info[0]  # Accede al diccionario correspondiente al único entorno
        # Determinar si el episodio fue truncado o terminó normalmente
        is_sucess = info.get("is_success")

        if done or (steps == args.eval_max_steps):
            print("rew = ", reward_sum)
            print("len = ", steps)
            # total reward per episode
            reward_array.append(reward_sum)
            # total steps taken per episode
            len_total.append(steps)
            if done and is_sucess:
                # Goal reached
                success.append(1)
            else:
                success.append(0)

            # Accede al entorno interno y se reinicia con la semilla
            if hasattr(env, 'envs') and len(env.envs) > 0:
                obs = env.envs[0].reset(seed=args.model_seed)
            else:
                obs = env.reset()
            # Extraer solo observaciones si env.reset() retorna un tuple
            if isinstance(obs, tuple):
                obs, _ = obs
            total += 1
            # reset variables
            reward_sum = 0
            steps = 0
        # evaluate n_eval_episodes
        if total == args.n_eval_episodes:
            break

    save_results(reward_array,len_total,success,args.eval_result_file+"_results_.csv")
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



def run_experiment_copilot(seed: int, alpha: float, is_pilot_noisy: bool, copilot_alpha: float):
    args = Params()

    if alpha is not None:
        args.alpha = alpha

    if copilot_alpha is not None:
        args.copilot_alpha = copilot_alpha

    if seed is not None:
        args.model_seed = seed

    # pilot_model is the bc_policy,
    # the model policy trained with imitation learning
    # the optimal model
    pilot_model = load_create_bc_model(args.pilot_model_path, args.log_path)

    if is_pilot_noisy:
        args.log_path = "logs-noisy-results/copilot_alpha_"+str(copilot_alpha)+"/"
        args.eval_result_file = os.path.join(args.log_path, "noisy-copilot-room1-results.csv")
        pilot = NoisyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)
    else:        
        args.log_path = "logs-laggy-results/copilot_alpha_"+str(copilot_alpha)+"/"
        args.eval_result_file = os.path.join(args.log_path, "laggy-copilot-room1-results.csv")
        pilot = LaggyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)

    # Inicializar BetaAdjuster
    # quitar pues no lo usamos
    beta_adjuster = BetaAdjuster(initial_beta=0.0, max_beta=1.0, total_episodes=args.model_total_episodes)
    # Initialize the environment
    env = CopilotCornerEnv(model=pilot_model,
                           pilot=pilot,
                           seed=args.model_seed,
                           alpha=args.copilot_alpha,
                           beta_adjuster=beta_adjuster)
    env.training_mode = False
    _, model = init_env_log_path(env, args.model_seed, pilot_model, args.log_path)
    # set the bc expert model to the copilot
    env.set_model(pilot_model)
    obs, _ = env.reset(seed=args.model_seed)
    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    success = []
    steps = 0

    # Evaluate the policy
    while True:
        # print("obs=", obs)
        action, _ = model.predict(obs,deterministic=True)
        # Convertir la acción a un array
        # if isinstance(action, (int, float)):
        #    action = np.array([action])
        # Validar que la acción sea compatible con el entorno
        #if len(action.shape) == 0:
        #    action = np.expand_dims(action, axis=0)
        print("Action:", action, "Action Shape:", action.shape)
        print("Observation:", obs, "Observation Shape:", obs.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Para mantener la lógica original
        print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward
        if isinstance(info, list):
            info = info[0]  # Accede al diccionario correspondiente al único entorno
        # Determinar si el episodio fue truncado o terminó normalmente
        is_sucess = info.get("is_success")

        if done or (steps == args.eval_max_steps):
            print("rew = ", reward_sum)
            print("len = ", steps)
            # total reward per episode
            reward_array.append(reward_sum)
            # total steps taken per episode
            len_total.append(steps)
            if done and is_sucess:
                # Goal reached
                success.append(1)
            else:
                success.append(0)

            # Accede al entorno interno y se reinicia con la semilla
            if hasattr(env, 'envs') and len(env.envs) > 0:
                obs = env.envs[0].reset(seed=args.model_seed)
            else:
                obs = env.reset()
            # Extraer solo observaciones si env.reset() retorna un tuple
            if isinstance(obs, tuple):
                obs, _ = obs
            total += 1
            # reset variables
            reward_sum = 0
            steps = 0
        # evaluate n_eval_episodes
        if total == args.n_eval_episodes:
            break

    save_results(reward_array,len_total,success,args.eval_result_file+"_results_.csv")
    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")

if __name__ == '__main__':

    # Laggy Pilot 0.25
    run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.01)
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.025)
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.05)
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=1)
    # Laggy Pilot 0.8
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.8, copilot_alpha=0.01)
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.8, copilot_alpha=0.025)
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.8, copilot_alpha=0.05)
    # run_experiment_copilot(is_pilot_noisy=False, seed=5, alpha=0.8, copilot_alpha=1)
    
    # Noisy Pilot
    # run_experiment_copilot(is_pilot_noisy=True, seed=5, alpha=0.25, copilot_alpha=0.01)
    # run_experiment_copilot(is_pilot_noisy=True, seed=5, alpha=0.25, copilot_alpha=0.025)
    # run_experiment_copilot(is_pilot_noisy=True, seed=5, alpha=0.25, copilot_alpha=0.05)
    # run_experiment_copilot(is_pilot_noisy=True, seed=5, alpha=0.25, copilot_alpha=1)
    
    # generate a Trajectories dataset
    # run_experiment_generate_trajectories_for_laggy_pilot()
    # 2025-02-10
    # 1) laggy train
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.5, want_to_train=True)
    # 2) test laggy
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.1, want_to_train=False)
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.25, want_to_train=False)
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.5, want_to_train=False)
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=0.75, want_to_train=False)
    # run_experiment(is_pilot_noisy=False, seed=5, alpha=0.25, copilot_alpha=1, want_to_train=False)
    
    # 4) noisy train
    # run_experiment(is_pilot_noisy=True, seed=5, alpha=0.3, copilot_alpha=0.5, want_to_train= True)

"""
Script to evaluate laggy pilot and noisy pilot
"""
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from stable_baselines3.common.callbacks import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from bc_expert.bc_expert import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *
from pilots.LaggyPilot import *
from pilots.NoisyPilot import *
from pilots.pilot import *


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    model_total_timesteps = 40_000
    model_verbose = True
    log_path = ""
    pilot_model_path = "../bc_expert/imitation/logs/bc_policy"
    n_eval_episodes = 10
    eval_result_file = ""


def run_experiment(p:Pilot, args:Params):
    # Initialize the environment
    #env = CornerEnv()
    env = SimpleCornerEnvRS10()
    env.set_trajectory_path(args.log_path)
    env = Monitor(env, filename=args.eval_result_file, info_keywords=("is_success",))
    env = DummyVecEnv([lambda: env])

    # Accede al entorno interno y reinícialo con la semilla
    if hasattr(env, 'envs') and len(env.envs) > 0:
        obs = env.envs[0].reset(seed=args.model_seed)
    else:
        obs = env.reset()
    # Extraer solo observaciones si env.reset() retorna un tuple
    if isinstance(obs, tuple):
        obs, _ = obs

    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    success = []
    steps = 0

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Evaluate the policy
    while True:
        # print("obs=", obs)
        action, _ = p.choose_action(obs)
        # Convertir la acción a un array
        if isinstance(action, (int, float)):
            action = np.array([action])
        # Validar que la acción sea compatible con el entorno
        if len(action.shape) == 0:
            action = np.expand_dims(action, axis=0)
        #print("Action:", action, "Action Shape:", action.shape)
        #print("Observation:", obs, "Observation Shape:", obs.shape)
        obs, reward, done, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward
        if isinstance(info, list):
            info = info[0]  # Accede al diccionario correspondiente al único entorno
        # Determinar si el episodio fue truncado o terminó normalmente
        is_sucess = info.get("is_success")

        if done or (steps == args.model_total_timesteps):
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

            # Accede al entorno interno y reinícialo con la semilla
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


    data = {'Reward': reward_array, 'Len': len_total,'Sucess': success}
    # Create DataFrame
    df = pd.DataFrame(data)
    print("Results")
    print(df)
    # Save DataFrame to CSV
    df.to_csv(args.eval_result_file+"_results_.csv", index=False)

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")

def test_pilot(seed: float, alpha: float, is_pilot_noisy: bool):
    args = Params()
    # args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    # load_create_bc_model(logdir=args.log_path)

    # pilot_model = PPO.load(args.pilot_model_path)
    # Configuración adicional
    if alpha is not None:
        args.alpha = alpha  # Coeficiente para noisy o laggy
    else:
        args.alpha = 0.5
    if seed is not None:
        args.model_seed = seed

    # Configura valores iniciales para log_path y eval_result_file
    if is_pilot_noisy:
        args.log_path = "./logs_noisy_test/"+str(int(alpha * 100))+"/"
        args.eval_result_file = os.path.join(args.log_path, "noisy_pilot_room1_"+str(int(alpha * 100))+"_.csv")
    else:
        args.log_path = "./logs_laggy_test/"+str(int(alpha * 100))+"/"
        args.eval_result_file = os.path.join(args.log_path, "laggy_pilot_room1_"+str(int(alpha * 100))+"_.csv")

    # Crea el directorio si no existe
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Carga el modelo
    pilot_model = load_create_bc_model(args.pilot_model_path, args.log_path)



    # Selecciona el piloto
    if is_pilot_noisy:
        pilot = NoisyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)
    else:
        pilot = LaggyPilot(model=pilot_model, seed=args.model_seed, alpha=args.alpha)

    # Corre el experimento
    run_experiment(pilot, args)


if __name__ == '__main__':
    # to evaluate a Noisy pilot, uncomment the following line & execute
    # test_pilot(5, 0.5, is_pilot_noisy=True)
    # to evaluate a Laggy pilot, uncomment the following line & execute
    # test_pilot(5, 0.5, is_pilot_noisy=False)
    # test_pilot(5, 0.75, is_pilot_noisy=False)
    # --------------------------------------
    # test_pilot(5, 0.25, is_pilot_noisy=True)
    # test_pilot(5, 0.5, is_pilot_noisy=True)
    test_pilot(5, 0.1, is_pilot_noisy=True)
    # test_pilot(5, 0.75, is_pilot_noisy=True)


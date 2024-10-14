"""
Test a laggy base-pilot in room 1
"""
import sys
import os
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.monitor import Monitor
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *

from controller import Keyboard
import numpy as np
from imitation.data.types import Trajectory


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    model_total_timesteps = 10_000
    log_path = "./logs_2024-10-13/"
    # increase this number later
    n_eval_episodes = 20
    eval_result_file = os.path.join(log_path, "trajectories-summary-results.csv")


def save_results(r, l, g, file_name):
    data = {'Reward': r, 'Len': l, 'Goal': g}
    # Create DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False, mode='w')


def is_control_key(key):
    # Definición de los códigos de teclas que queremos identificar
    UP_KEY = Keyboard.UP
    RIGHT_KEY = Keyboard.RIGHT
    DOWN_KEY = Keyboard.DOWN
    LEFT_KEY = Keyboard.LEFT
    Q_KEY = ord('Q')
    W_KEY = ord('E')
    Y_KEY = ord('Y')

    # Verifica si la tecla ingresada coincide con alguna de las especificadas
    return key in {UP_KEY, RIGHT_KEY, DOWN_KEY, LEFT_KEY, Q_KEY, W_KEY, Y_KEY}


def get_action_from_user(key):
    action = 0
    if key == Keyboard.UP:
        action = 0
    elif key == Keyboard.DOWN:
        action = 1
    elif key == Keyboard.RIGHT:
        action = 2
    elif key == Keyboard.LEFT:
        action = 3
    elif key == ord('Q'):
        action = 4
    elif key == ord('E'):
        action = 5

    return key


def run_experiment(want_to_train=True):
    args = Params()
    # Initialize the environment
    env = CornerEnv()
    # https://github.com/openai/gym/issues/681
    #rng = np.random.default_rng(args.model_seed)
    #env.action_space.np_random = rng
    #env.action_space.np_random.seed(args.model_seed)
    env.seed(args.model_seed)

    keyboard = env.get_webots_keyboard()

    env = Monitor(env, filename=args.log_path, info_keywords=("is_success", "corner",))

    #env.get_wrapper_attr('seed')(args.model_seed)
    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    goal = []
    steps = 0
    obs = env.reset()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    trajectories = []
    observations = []
    actions = []
    rewards = []
    infos = []
    terminals = []

    print("====== Controls =======\n\n")
    print(" The Crazyflie can be controlled from your keyboard!")
    print(" All controllable movement is in body coordinates")
    print("- Use the up, back, right and left button to move in the horizontal plane")
    print("- Use Q and E to rotate around yaw ")
    print("- Use Y to stop the experiment")
    # print("- Use W and S to go up and down ")

    while True:
        key = keyboard.getKey()
        if key != -1:
            if is_control_key(key):
                if key == ord('Y'):
                    break

                action = get_action_from_user(key)
                obs, reward, done, truncated, info = env.step(action)
                print("obs=", obs, "reward=", reward, "done=", done)
                observations.append(obs)
                actions.append(action)
                rewards.append(info)
                infos.append(info)
                steps = steps + 1
                reward_sum = reward_sum + reward
                # if an episode ends
                if done or args.model_total_timesteps < steps:
                    if done:
                        goal.append(True)
                        print("Goal reached! rew = ", reward_sum)
                        print("Goal reached! len = ", steps)
                        terminals.append(True)
                    else:
                        goal.append(False)
                        terminals.append(True)

                    # Crear un objeto Trajectory con recompensas
                    trajectory = Trajectory(
                        obs=observations,
                        acts=actions,
                        infos=infos,
                        rews=rewards,
                        dones=terminals
                    )
                    trajectories.append(trajectory)
                    # new episode
                    obs = env.reset()
                    total += 1
                    # reset variables
                    reward_sum = 0
                    steps = 0
                    observations = []
                    actions = []
                    rewards = []
                    infos = []
                    terminals = []

        if total == args.n_eval_episodes:
            break

    # Guardar el dataset con varios episodios
    np.savez("multi_episode_dataset.npz", **trajectories)
    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")


if __name__ == '__main__':
    run_experiment()

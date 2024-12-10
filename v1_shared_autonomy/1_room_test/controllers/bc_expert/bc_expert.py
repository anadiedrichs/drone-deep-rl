import sys

import pandas as pd
import torch
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from imitation.algorithms.bc import BC
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *

"""
Test an expert trained with BC in room 1
"""

class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    model_total_timesteps = 50_000
    log_path = "./logs_20241208_det_False/"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "baseline-random-results.csv")

def save_results(r,l,g,file_name):
    data = {'Reward': r, 'Len': l, 'Goal':g}
    # Create DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False,mode='w')

def load_create_bc_model(model_path=None,logdir=None):

    print(model_path)
    print(logdir)
    # setup CornerEnv observations & action space
    o_s = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    a_s = Discrete(6)
    # Create a random number generator
    rng = np.random.default_rng(0)
    # logs setup
    # logs setup
    log_dir = logdir if logdir is not None else "./bc_evaluate"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

    # trained policy file path
    model_path = model_path or "./imitation/logs/bc_policy"

    logger = configure(log_dir, ["tensorboard", "csv"])
    # Init BC
    bc_model = BC(observation_space=o_s, action_space=a_s,
                  demonstrations=None, rng=rng, custom_logger=logger)

    # load the trained policy model
    bc_model.policy.load_state_dict(torch.load(model_path))
    # Formato de la política entrenada
    print(bc_model.policy)
    return bc_model

def print_results(r,l,g):
    df = pd.DataFrame({'Reward': r, 'Len': l, 'Goal':g})
    print("Results")
    print(df)
def run_experiment(want_to_train=True):

    bc_model = load_create_bc_model()
    args = Params()
    # Initialize the environment
    env = SimpleCornerEnvRS10()
    env.set_trajectory_path(args.log_path)
    # https://github.com/openai/gym/issues/681
    # env.seed(args.model_seed)
    # Crear un entorno vectorizado
    env = Monitor(env, filename=args.log_path, info_keywords=env.get_info_keywords())
    env = DummyVecEnv([lambda: env])
    total = 0
    reward_sum = 0
    reward_array = []
    len_total = []
    goal=[]
    steps = 0
    obs= env.reset()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    save_results(reward_sum, len_total, goal, args.eval_result_file)

    while True:
        action = bc_model.policy.predict(obs, deterministic=False)
        # return obs, reward, done, self.truncated, info
        obs, reward, done, info = env.step(action)
        # print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward
        # if an episode ends
        if done or args.model_total_timesteps < steps:
            if done:
                goal.append(True)
                print("Goal reached! rew = ", reward_sum)
                print("Goal reached! len = ", steps)
            else:
                goal.append(False)
            # total reward per episode
            reward_array.append(reward_sum)
            # total steps taken per episode
            len_total.append(steps)
            save_results(reward_array,len_total,goal,args.eval_result_file)
            # new episode
            obs = env.reset()
            total += 1
            # reset variables
            reward_sum = 0
            steps = 0

        if total == args.n_eval_episodes:
            break

    print_results(reward_array,len_total,goal)
    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")


if __name__ == '__main__':

    run_experiment()

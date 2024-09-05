"""
Script to evaluate laggy pilot and noisy pilot
"""
import sys
import os
from stable_baselines3.common.callbacks import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CrazyflieDrone import CornerEnv
from pilots.LaggyPilot import *
from pilots.NoisyPilot import *
from pilots.pilot import *
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import pandas as pd


class Params:
    """
    Common model hyperparameters for PPO and
    other configs
    """
    model_seed = 5
    model_total_timesteps = 100_000
    model_verbose = True
    kl_target = 0.015
    #gae_gamma = 0.99
    gae_lambda = 0.95
    batch_size = 64
    n_steps = 512
    ls_rate = 0.001  # linear schedule rate
    log_path = "./logs_2024-09-04/"
    pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    # increase this number later
    n_eval_episodes = 10
    eval_result_file = os.path.join(log_path, "laggy-noisy-base-pilot-results.csv")


def run_experiment(p:Pilot, args:Params):
    # Initialize the environment
    env = CornerEnv()
    env = Monitor(env, filename=args.log_path, info_keywords=("is_success",))

    obs = env.reset()
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
        # print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)
        steps = steps + 1
        reward_sum = reward_sum + reward

        if done or (steps == args.model_total_timesteps):
            print("rew = ", reward_sum)
            print("len = ", steps)
            # total reward per episode
            reward_array.append(reward_sum)
            # total steps taken per episode
            len_total.append(steps)
            if done:
                # Goal reached
                success.append(1)
            else:
                success.append(0)

            # new episode
            obs = env.reset()
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
    df.to_csv(args.eval_result_file, index=False)

    # close Webots simulator
    # env.simulationQuit(0)
    print("run_experiment ended")

def test_pilot(seed: float, alpha: float, is_pilot_noisy: bool):
    args = Params()
    # args.pilot_model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"
    pilot_model = PPO.load(args.pilot_model_path)

    if alpha is not None:
        args.copilot_alpha = alpha # TODO this should be the coefficient for noisy or laggy
    if seed is not None:
        args.model_seed = seed

    if is_pilot_noisy:
        # where is the model to load
        args.log_path = "./logs_noisy_test/"
        args.eval_result_file = os.path.join(args.log_path, "noisy_pilot_room1_results.csv")
        pilot = NoisyPilot(pilot_model, args.model_seed)
    else:
        args.log_path = "./logs_laggy_test/"
        args.eval_result_file = os.path.join(args.log_path, "laggy_pilot_room1_results.csv")
        pilot = LaggyPilot(pilot_model, args.model_seed)

    run_experiment(pilot,args)

if __name__ == '__main__':

    test_pilot(5, 0.5, is_pilot_noisy=True)
    # test_pilot(5, 0.5, is_pilot_noisy=False)

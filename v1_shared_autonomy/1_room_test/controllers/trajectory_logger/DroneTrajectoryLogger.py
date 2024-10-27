
from stable_baselines3.common.monitor import Monitor
from imitation.data.types import Trajectory
from controller import Keyboard
from copilot.CornerEnv import *
import sys
import os
import numpy as np

class DroneTrajectoryLogger:
    def __init__(self, model_seed=5, model_total_timesteps=10000, log_path="./logs_2024-10-13/", n_eval_episodes=5):
        self.model_seed = model_seed
        self.model_total_timesteps = model_total_timesteps
        self.log_path = log_path
        self.n_eval_episodes = n_eval_episodes
        self.eval_result_file = os.path.join(log_path, "trajectories-summary-results.csv")

        # Environment setup
        self.env = None
        self.keyboard = None

        # Data collection
        self.trajectories = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.terminals = []

    def initialize_environment(self):
        self.env = CornerEnv()
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.env = Monitor(self.env, filename=self.log_path, info_keywords=self.env.get_info_keywords())
        self.keyboard = self.env.get_webots_keyboard()

    def print_menu(self):
        print("====== Controls =======\n\n")
        print(" The Crazyflie can be controlled from your keyboard!")
        print(" All controllable movement is in body coordinates")
        print("- Use the up, back, right and left button to move in the horizontal plane")
        print("- Use Q and E to rotate around yaw ")
        print("- Use X to stop the experiment")

    def is_control_key(self, key):
        UP_KEY = Keyboard.UP
        RIGHT_KEY = Keyboard.RIGHT
        DOWN_KEY = Keyboard.DOWN
        LEFT_KEY = Keyboard.LEFT
        Q_KEY = ord('Q')
        W_KEY = ord('E')
        Y_KEY = ord('X')
        return key in {UP_KEY, RIGHT_KEY, DOWN_KEY, LEFT_KEY, Q_KEY, W_KEY, Y_KEY}

    def get_action_from_user(self, key):
        action_mapping = {
            Keyboard.UP: 0,
            Keyboard.DOWN: 1,
            Keyboard.RIGHT: 2,
            Keyboard.LEFT: 3,
            ord('Q'): 4,
            ord('E'): 5,
        }
        return action_mapping.get(key, 0)

    def run_experiment(self):
        self.initialize_environment()
        self.print_menu()
        obs = self.env.reset()
        total_episodes = 0
        reward_sum = 0
        steps = 0

        while True:
            key = self.keyboard.getKey()
            if key != -1 and self.is_control_key(key):
                if key == ord('X'):
                    break

                action = self.get_action_from_user(key)
                obs, reward, done, truncated, info = self.env.step(action)
                self.observations.append(obs)
                self.rewards.append(reward)
                self.terminals.append(done)
                self.infos.append(info)
                self.actions.append(action)
                steps += 1
                reward_sum += reward
                if done or steps >= self.model_total_timesteps:
                    self.save_trajectory(done, reward_sum, steps)
                    if total_episodes >= self.n_eval_episodes:
                        break
                    obs = self.env.reset()
                    total_episodes += 1
                    reward_sum, steps = 0, 0

        # Save the dataset
        self.save_dataset()
        print("Experiment ended")

    def save_trajectory(self, goal_reached, reward_sum, steps):
        goal = True if goal_reached else False
        self.terminals.append(goal)
        trajectory = Trajectory(
            obs=self.observations,
            acts=self.actions,
            infos=self.infos,
            rews=self.rewards,
            dones=self.terminals,
        )
        self.trajectories.append(trajectory)
        self.observations, self.actions, self.rewards, self.infos, self.terminals = [], [], [], [], []

    def save_dataset(self):
        trajectory_dict = [trajectory._asdict() for trajectory in self.trajectories]
        np.savez("multi_episode_dataset.npz", *trajectory_dict)



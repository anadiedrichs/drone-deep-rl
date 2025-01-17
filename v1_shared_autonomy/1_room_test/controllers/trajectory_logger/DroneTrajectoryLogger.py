import os
import sys
import numpy as np
import time
import threading
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from imitation.data.types import Trajectory
from controller import Keyboard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from copilot.CornerEnv import *
from pilots.LaggyPilot import *
from pilots.NoisyPilot import *
from copilot.Copilots import CopilotCornerEnv

class DroneTrajectoryLogger:
    def __init__(self, use_copilot=False, pilot_type=None, model=None, 
                 model_seed=5, model_total_timesteps=10000, 
                 log_path="./logs__/", n_eval_episodes=5,
                 pilot_alpha=0.5, copilot_alpha=0.5):
        """
        Initializes the DroneTrajectoryLogger.
        Args:
        use_copilot (bool): If True, use CopilotCornerEnv; otherwise, use SimpleCornerEnvRS10.
        pilot_type (str): Type of pilot for CopilotCornerEnv. Options are 'Laggy' or 'Noisy'.
        model (PPO or None): The PPO model for the copilot's policy.
        model_seed (int): Seed for the model.
        model_total_timesteps (int): Total timesteps for the experiment.
        log_path (str): Path to store logs and trajectories.
        n_eval_episodes (int): Number of episodes to evaluate.
        pilot_alpha (float): Alpha value for the pilot's behavior.
        copilot_alpha (float): Alpha value for the copilot's behavior.
        """
        self.use_copilot = use_copilot
        self.pilot_type = pilot_type
        self.model = model
        self.pilot_alpha = pilot_alpha
        self.copilot_alpha = copilot_alpha
        self.model_seed = model_seed
        self.model_total_timesteps = model_total_timesteps
        self.log_path = log_path
        self.n_eval_episodes = n_eval_episodes
        self.eval_result_file = os.path.join(log_path, "trajectories-summary-results.csv")

        # Environment setup
        self.env = None
        self.keyboard = None
        self.current_key = None
        # Data collection
        self.trajectories = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.terminals = []

    def initialize_environment(self):
        """
        Initializes the environment based on the use_copilot flag.
        """
        if self.use_copilot:
            # Use CopilotCornerEnv
            if self.pilot_type == 'Laggy':
                pilot = LaggyPilot(self.model, seed=self.model_seed, alpha=self.pilot_alpha)
            elif self.pilot_type == 'Noisy':
                pilot = NoisyPilot(self.model, seed=self.model_seed, alpha=self.pilot_alpha)
            else:
                raise ValueError("Invalid pilot_type. Choose 'Laggy' or 'Noisy'.")
            self.env = CopilotCornerEnv(model=self.model, pilot=pilot, seed=self.model_seed, alpha=self.copilot_alpha)
        else:
            # Use SimpleCornerEnvRS10 to gather data from human pilots
            self.env = SimpleCornerEnvRS10()

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.keyboard = self.env.get_webots_keyboard()
        if self.keyboard is not None:
            self.keyboard.enable(self.env.timestep)
        else:
            raise RuntimeError("Keyboard could not be initialized.")

        self.env.set_trajectory_path(self.log_path)
        self.env = Monitor(self.env, filename=self.log_path, info_keywords=self.env.get_info_keywords())

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
            Keyboard.LEFT: 2,
            Keyboard.RIGHT: 3,
            ord('Q'): 4,
            ord('E'): 5,
        }
        return action_mapping.get(key, 0)

    def run_experiment_key(self):
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

    def save_trajectory(self, goal_reached):
        goal = True if goal_reached else False
        self.terminals.append(goal)
        trajectory = {
            "obs": np.array(self.observations),
            "acts": np.array(self.actions),
            "rews": np.array(self.rewards),
            "infos": self.infos,
            "dones": self.terminals,
        }
        self.trajectories.append(trajectory)
        self.observations, self.actions, self.rewards, self.infos, self.terminals = [], [], [], [], []

    def save_dataset(self,file_name):
        #trajectory_dict = [trajectory._asdict() for trajectory in self.trajectories]
        np.savez(self.log_path+file_name+".npz", trajectories=self.trajectories)
    
    def save_results(self,file_name):
        df = pd.DataFrame({
            "obs": np.array(self.observations),
            "acts": np.array(self.actions),
            "rews": np.array(self.rewards),
            "infos": self.infos,
            "dones": self.terminals,
        })
        df.to_csv(self.log_path+file_name+".csv", index=False)

    def run_experiment_wo_keyboard_test(self):
        self.initialize_environment()
        self.print_menu()
        obs = self.env.reset()
        total_episodes = 0
        reward_sum = 0
        steps = 0
        simulated_actions = [0, 2, 3, 1, 4, 5]  # Lista de acciones simuladas

        for action in simulated_actions:
            obs, reward, done, truncated, info = self.env.step(action)

            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.infos.append(info)
            self.terminals.append(done)

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

    def keyboard_listener(self):
        while True:
            key = self.keyboard.getKey()
            #print(f"Key detected: {key}")
            if key != -1:
                self.current_key = key

    def run_experiment_human(self,file_name,debug=True):
        self.initialize_environment()

        if debug:
            print("Key 1:",str(self.current_key))

        obs = self.env.reset()

        if debug:
            print("Key after reset:",str(self.current_key))

        threading.Thread(target=self.keyboard_listener, daemon=True).start()
        self.print_menu()

        total_episodes = 1
        reward_sum = 0
        steps = 0

        while True:

            if self.current_key != -1 and self.is_control_key(self.current_key):
                if self.current_key == ord('Y'):
                    break

                action = self.get_action_from_user(self.current_key)
                obs, reward, done, truncated, info = self.env.step(action)
                if debug:
                    print(f"Key: {self.current_key}, Obs: {obs}, Reward: {reward}, Done: {done}")

                self.observations.append(obs)
                self.actions.append(action)
                self.rewards.append(reward)
                self.infos.append(info)
                self.terminals.append(done)

                steps += 1
                reward_sum += reward

                if done or steps >= self.model_total_timesteps:
                    self.save_trajectory(done)
                    if total_episodes >= self.n_eval_episodes:
                        break
                    obs = self.env.reset()
                    total_episodes += 1
                    reward_sum, steps = 0, 0
            else:
                obs, reward, done, truncated, info = self.env.step(-2)
                #if debug:
                #    print(f"Key: {self.current_key}, Obs: {obs}, Reward: {reward}, Done: {done}")

        # Save the dataset
        self.save_dataset(file_name)
        print("Experiment ended")

    def run_experiment_copilot(self,file_name,debug=True, train= True):

        self.initialize_environment()
        self.env.training_mode = train
        obs, _ = self.env.reset()
        total_episodes = 1
        reward_sum = 0
        steps = 0

        while True:

            action = self.env.get_action(obs)
            # Convertir la acción a un array
            if isinstance(action, (int, float)):
                action = np.array([action])
            # Validar que la acción sea compatible con el entorno
            if len(action.shape) == 0:
                action = np.expand_dims(action, axis=0)
            #print("Action:", action, "Action Shape:", action.shape)
            #print("Observation:", obs, "Observation Shape:", obs.shape)

            obs, reward, done, truncated, info = self.env.step(action)
            if debug:
                print(f"Key: {self.current_key}, Obs: {obs}, Reward: {reward}, Done: {done}")

            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.infos.append(info)
            self.terminals.append(done)

            steps += 1
            reward_sum += reward

            if done or steps >= self.model_total_timesteps:
                self.save_trajectory(done)
                if total_episodes >= self.n_eval_episodes:
                    break
                obs = self.env.reset()
                total_episodes += 1
                reward_sum, steps = 0, 0

        # Save the dataset needed for imitation learning
        # self.env.training_mode
        self.save_dataset(file_name)
        self.save_results(file_name)
        print("Experiment ended")



    def run_experiment(self, file_name,debug=True, train = True):

        if file_name is None:
            file_name = "output_file"

        if self.pilot_type is None:
            # colect data from a human pilot who drives the drone
            self.run_experiment_human(file_name, debug)
        else:
            # colect data from the copilot in training or testing mode 
            self.run_experiment_copilot(file_name, debug, train)

        
"""
More runners for discrete RL algorithms can be added here.
"""

import sys
from controller import Supervisor
from controller import Keyboard # user input
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Camera
from controller import DistanceSensor
from math import cos, sin, pi
from gym.spaces import Box, Discrete


sys.path.append('../../../utils')
from utilities import *
from pid_controller import *


sys.path.append('../../../copilot')
from CrazyflieDrone import *


MAX_HEIGHT = 0.7 # in meters
HEIGHT_INCREASE = 0.05 # 5 cm
HEIGHT_INITIAL = 0.30 # 20 cm
WAITING_TIME = 5 # in seconds 
# 2000 mm es el máximo, 200 mm = 20 cm  
DIST_MIN = 1000 # in mm (200 + 200 + 1800 + 1800 )/4

try:
    import gym #nasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym==0.21 stable_baselines3"'
    )

from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

class PilotRoom1(DroneOpenAIGymEnvironment):

    def __init__(self):
        super().__init__()

    def get_reward(self, action=6):
        
        """
                
        """
        reward = 0 
        action = int(action)
        r_avoid_obstacle = 0

        # 2000 mm es el máximo, 100 mm = 10 cm
        # if the drone reach a corner
        if bool((self.dist_front <= 100 and self.dist_left <= 100) or \
                (self.dist_front <= 100 and self.dist_right <= 100) or \
                (self.dist_back <= 100 and self.dist_left <= 100) or \
                (self.dist_back <= 100 and self.dist_right <= 100)):

            # drone wins a big reward
            reward = 10
        else:

            # be near an obstacle
            too_close = (self.dist_front <= 100 or self.dist_left <= 100 or \
                   self.dist_right <= 100 or self.dist_back <= 100 )

            if too_close:
                # penalize to be too close an obstacle or wall
                r_avoid_obstacle = -1
            else:
                # reward to keep going
                r_avoid_obstacle = 1

        reward =reward+r_avoid_obstacle

        # Calcular valores mínimo y máximo de recompensa
        # Dependiente del escenario
        min_reward = -1 #
        max_reward = 10 #

        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, min_reward, max_reward, -1, 1)
    
       
        # Reward for every step the episode hasn't ended
        print("DEBUG Reward value " + str(reward))
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward


    def is_done(self):
        """
        Return True when:
         * the drone reach a corner
        """
        
        # if the drone reach a corner
        if bool((self.dist_front <= 100 and self.dist_left <= 100) or \
                (self.dist_front <= 100 and self.dist_right <= 100) or \
                (self.dist_back <= 100 and self.dist_left <= 100) or \
                (self.dist_back <= 100 and self.dist_right <= 100)):

            return True

            
        return False
            
def train_pilot():
    # Initialize the environment
    env = PilotRoom1()
    #check_env(env)

    tmp_path = "./logs/"
    # set up logger
    # https://stable-baselines3.readthedocs.io/en/master/common/logger.html#logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    # Train
    # PPO Proximal Policy Optimization (PPO)
    model = PPO('MlpPolicy', env, n_steps=1024,
                verbose=1,
                tensorboard_log=tmp_path,
                seed=7)
    # Set new logger
    model.set_logger(new_logger)

    model.learn(total_timesteps=1000)
    
    # Save the model
    model.save("ppo_model")

    obs = env.reset()
    
    
#    for i in range(100000):
#        print("i : " + str(i) )
#        action, _states = model.predict(obs)
#        obs, reward, done, info = env.step(action)
#        print(obs, reward, done, info)
#        if done:
            #print("DONE is TRUE ")
#            obs = env.reset()


    # Evaluate the trained agent
#    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    
#    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
if __name__ == '__main__':
    train_pilot()
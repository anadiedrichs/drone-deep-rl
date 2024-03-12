from numpy import convolve, ones, mean
import gym

from my_controller import DroneRobotSupervisor
from utilities import plot_data

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env


def run():
    # Initialize supervisor object
    env = DroneRobotSupervisor()
    
    # Verify that the environment is working as a gym-style env
    #check_env(env)
    
    # The drone takes off, then execute random actions
    while True:       
       # Random action
        action = env.action_space.sample()
        print("ACTION "+str(action))
        
        #print ('Press 1 if the drone took off')
        key=env.keyboard.getKey()
        
        if (key==ord('1')):
            print ('You pressed 1')
            break
        if env.the_drone_took_off:
            print("DESPEGOOOOOOOOOOOOOOOOOO")
        
        obs, reward, is_done, info = env.step(action)
        
        if is_done:
            obs, info = env.reset()
            print("It's done !")
  
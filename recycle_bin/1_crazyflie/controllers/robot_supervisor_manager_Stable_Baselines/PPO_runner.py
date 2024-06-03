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
    # check_env(env)
        
    # despegar drone
    env.take_off()
    
    #obs, info = env.reset()
    n_steps = 10
    
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        print("ACTION "+str(action))
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if done:
            obs, info = env.reset()
    
    if(False):
        # Verify that the environment is working as a gym-style env
        # check_env(env)
        
        #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
        model = PPO("MlpPolicy", env, verbose=1)
        # Indicate the total timmepstes that the agent should be trained.
        model.learn(total_timesteps=4000)
        # Save the model
        model.save("ppo_model")
    
        #del model # remove to demonstrate saving and loading
    
        # If is needed to load the trained model
        #model = PPO.load("ppo_model")
    
        ################################################################
        # End of the training period and now we evaluate the trained agent
        ################################################################
        
        # Initialize the environment
        obs = env.reset()
        env.episode_score = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            env.episode_score += reward  # Accumulate episode reward
    
            if done:
                print("Reward accumulated =", env.episode_score)
                env.episode_score = 0
                obs = env.reset()
    

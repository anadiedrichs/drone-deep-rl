from numpy import convolve, ones, mean
import gym

from my_controller import DroneRobotSupervisor
from utilities import plot_data

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import  check_env

from stable_baselines3.common.logger import configure

tmp_path = "/tmp/sb3_log/"
# set up logger
new_logger = configure(tmp_path, [ "csv", "tensorboard"])

def run():
    # Initialize supervisor object
    env = DroneRobotSupervisor()
    
    key=env.keyboard.getKey()
    if (key==ord('1')):
        print ('You pressed 1')
        
    if env.the_drone_took_off:
        print("DESPEGOOOOOOOOOOOOOOOOOO")
    # Verify that the environment is working as a gym-style env
    #check_env(env)
    print(" Environment created " ) 
    
    #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
    model = DQN("MlpPolicy", env, verbose=1,seed=7)
    model.set_logger(new_logger)
    # Indicate the total timmepsteps that the agent should be trained.
    model.learn(total_timesteps=40000)
    # Save the model
    model.save("ppo_model")

    #del model # remove to demonstrate saving and loading

    # If is needed to load the trained model
    #model = PPO.load("ppo_model")

    ################################################################
    # End of the training period and now we evaluate the trained agent
    ################################################################
    
    if False:
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
            
    
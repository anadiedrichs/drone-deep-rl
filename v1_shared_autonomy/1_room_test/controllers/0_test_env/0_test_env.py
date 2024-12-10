"""

Create a custom environment.

Then we test the different methods

"""

import sys
from controller import Supervisor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *


from stable_baselines3.common.env_checker import check_env

class PilotRoom1(DroneRobotSupervisor):

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
            
def test_env():

    # Initialize the environment
    env = PilotRoom1()

    check_env(env)

    # Box(4,) means that it is a Vector with 4 components
    print("Observation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    # Discrete(2) means that there is two discrete actions
    print("Action space:", env.action_space)

    # The reset method is called at the beginning of an episode
    obs = env.reset()
    # Sample a random action
    action = env.action_space.sample()
    print("Sampled action:", action)
    obs, reward, terminated, info = env.step(action)
    # Note the obs is a numpy array
    # info is an empty dict for now but can contain any debugging info
    # reward is a scalar
    print(obs.shape, reward, terminated, info)


    while env.keyboard.getKey() != ord('Y'):
        print("press Y to close the simulator")
        env.step(action)

    # close Webots simulator
    env.simulationQuit(0)

class Params13:

    model_seed = 7
    max_episode_steps=20_000
    model_total_timesteps = 100_000
    model_verbose = True
    log_path = "./20241208_RS10_sb3algos/"
    tb_log_name = "20241208"

def test_SimpleCornerEnvRS10():
    args = Params13()
    model = None
    # Initialize the environment
    env = SimpleCornerEnvRS10(args.max_episode_steps)

    print("Tipee Y para confirmar el experimento")
    env.wait_keyboard()

    k= env.get_webots_keyboard()
if __name__ == '__main__':
    test_env()
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
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor
from math import cos, sin, pi
from gym.spaces import Box, Discrete
import numpy as np


from utilities import *
from pid_controller import pid_velocity_fixed_height_controller


MAX_HEIGHT = 1 # in meters
HEIGHT_INCREASE = 0.05 # 1 cm
HEIGHT_INITIAL = 0.3 # 10 cm


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


class DroneOpenAIGymEnvironment(Supervisor, gym.Env):

    def __init__(self, max_episode_steps=1000):
        super().__init__()
        
        # 1) OpenAIGym generics
         # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        
        #[roll, pitch, yaw_rate, v_x, v_y, self.altitude ,self.range_front_value, self.range_back_value ,self.range_right_value, self.range_left_value ] #4
        
        self.observation_space = Box(low=np.array([- pi, -pi/2, - pi, 0,0, 0.1, 2.0, 2.0, 2.0, 2.0]),
                                     high=np.array([pi, pi/2, pi, 10, 10, 5,2000, 2000, 2000, 2000]),
                                     dtype=np.float64)
        
        # Define agent's action space using Gym's Discrete
        
        self.action_space = Discrete(7) 
            
        self.spec = gym.envs.registration.EnvSpec(id='DroneWebotsEnv-v0', max_episode_steps=max_episode_steps)

        # 2) Environment specific configuration 
        
        timestep = int(self.getBasicTimeStep())
        self.timestep = timestep
             
        # Crazyflie velocity PID controller
        self.PID_crazyflie = None
         
        # Initialize Sensors
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/inertialunit.md
        self.imu = None
        self.gps = None
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/gyro.md
        self.gyro = None
        self.camera = None
        self.range_front = None
        self.range_left = None
        self.range_back = None
        self.range_right = None               
            
            
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.altitude = 0 # alt normalized 
        self.x_global = 0.0
        self.y_global = 0.0
        self.first_time = True
        self.dt = 1.0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.yaw_rate = 0
        self.v_x = 0
        self.v_y = 0
        self.alt = 0 # in meters         
        self.past_time = 0
        self.the_drone_took_off = False
        self.height_desired = HEIGHT_INITIAL
        self.timestamp_take_off = 0
        
        # Initialize motors
        self.motors = None
        self.initialization() 
                
       
        print("DEBUG init DroneRobotSupervisor")
        
    def initialization(self):
        """ Internal function to initializse sensors and actuators"""
                
        self.timestep = int(self.getBasicTimeStep())
             
        # Crazyflie velocity PID controller
        self.PID_crazyflie = pid_velocity_fixed_height_controller()
         
        # Initialize Sensors
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/inertialunit.md
        self.imu = self.getDevice("inertial_unit")
        self.imu.enable(self.timestep)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/gyro.md
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.timestep)
        self.camera = self.getDevice("camera") # not used
        self.camera.enable(self.timestep)
        self.range_front = self.getDevice("range_front")
        self.range_front.enable(self.timestep)
        self.range_left = self.getDevice("range_left")
        self.range_left.enable(self.timestep)
        self.range_back = self.getDevice("range_back")
        self.range_back.enable(self.timestep)
        self.range_right = self.getDevice("range_right")
        self.range_right.enable(self.timestep)
        
        # Get keyboard, not currently in use
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)
            
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.altitude = 0 # alt normalized 
        self.x_global = 0.0
        self.y_global = 0.0
        self.first_time = True
        self.dt = 1.0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.yaw_rate = 0
        self.v_x = 0
        self.v_y = 0
        self.alt = 0 # in meters         
        self.past_time = 0
        self.the_drone_took_off = False
        self.height_desired = HEIGHT_INITIAL
        self.timestamp_take_off = 0
        # Initialize motors
        self.motors = [None for _ in range(4)]
        self.setup_motors() 
        
       
        print("DEBUG initialization")
        
    def setup_motors(self):
        """
        This method initializes the four motors.
        """
        
        self.motors[0] = self.getDevice('m1_motor')
        self.motors[1] = self.getDevice('m2_motor')
        self.motors[2] = self.getDevice('m3_motor')
        self.motors[3] = self.getDevice('m4_motor')
        
        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))
            
        self.setup_motors_velocity(np.array([1.0,1.0,1.0,1.0]))
      
    def setup_motors_velocity(self,motor_power):
        """
        This method sets the motors' velocity
        """
       
        self.motors[0].setVelocity(-motor_power[0])
        self.motors[1].setVelocity(motor_power[1])
        self.motors[2].setVelocity(-motor_power[2])
        self.motors[3].setVelocity(motor_power[3])
        print("====== Motors velocity =======\n")
        print(" m1 " + str(-motor_power[0]) ) # 1
        print(" m2 " + str(motor_power[1]) )  # 2
        print(" m3 " + str(-motor_power[2]) ) # 3       
        print(" m4 " + str(motor_power[3]) )  # 4
        
    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.timestep)
    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)

        # Motors & sensors
        self.initialization()

        # Internals
        super().step(self.timestep)
        
        self.take_off()

        # Open AI Gym generic
        return self.get_default_observation()
        
    def get_observations(self):
    
        
        if self.first_time:
            self.past_x_global = self.gps.getValues()[0]
            self.past_y_global = self.gps.getValues()[1]
            self.past_time = self.getTime()
            self.first_time = False
        
        self.dt = self.getTime() - self.past_time
            
        #print("dt   " + str(self.dt) )
        #print("past_time   " + str(self.past_time) )
        #print("getTime   " + str(self.getTime()) )
        if self.dt == 0 : # solve division by zero bug
            self.dt = self.timestep

        # Get sensor data
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/inertialunit.md
        self.roll = self.imu.getRollPitchYaw()[0]
        self.pitch = self.imu.getRollPitchYaw()[1]
        self.yaw = self.imu.getRollPitchYaw()[2]
        # The angular velocity is measured in radians per second [rad/s].c
        self.yaw_rate = self.gyro.getValues()[2] 
        self.x_global = self.gps.getValues()[0]
        v_x_global = (self.x_global - self.past_x_global)/self.dt
        self.y_global = self.gps.getValues()[1]
        v_y_global = (self.y_global - self.past_y_global)/self.dt
        
        self.alt = self.gps.getValues()[2]
        
        # Get body fixed velocities
        cos_yaw = cos(self.yaw)
        sin_yaw = sin(self.yaw)
        self.v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        self.v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # camera_data = camera.getImage()

        # to get the value in meters you have to divide by 1000
        self.dist_front = self.range_front.getValue()
        self.dist_back = self.range_back.getValue()
        self.dist_right = self.range_right.getValue()
        self.dist_left = self.range_left.getValue()
        
        # Normalize values for RL model 
        roll = normalize_to_range(self.roll,- pi, pi, -1.0, 1.0,clip=True)
        pitch = normalize_to_range(self.pitch,- pi/2, pi/2, -1.0, 1.0,clip=True)
        yaw_rate = normalize_to_range(self.yaw_rate,- pi, pi, -1.0, 1.0,clip=True)
        v_x = normalize_to_range(self.v_x,0, 10, -1.0, 1.0,clip=True)
        v_y = normalize_to_range(self.v_y,0, 10, -1.0, 1.0,clip=True)
        self.altitude = normalize_to_range(self.alt ,0.1, 5, -1.0, 1.0,clip=True)
        range_front_value = normalize_to_range(self.dist_front,2, 2000, -1.0, 1.0,clip=True)
        range_back_value = normalize_to_range(self.dist_back,2, 2000, -1.0, 1.0,clip=True)
        range_right_value = normalize_to_range(self.dist_right,2, 2000, -1.0, 1.0,clip=True)
        range_left_value = normalize_to_range(self.dist_left,2, 2000, -1.0, 1.0,clip=True)
    
       
        print("====== SENSORS observations =======\n")
        print("dt   " + str(self.dt) )
        print("Roll   " + str(self.roll) )
        print("Pitch  " + str(self.pitch) )
        print("Yaw    " + str(self.yaw) )        
        print("Yaw rate: " + str(self.yaw_rate) )        
        print("x_global: " + str(self.x_global) )
        print("v_x_global: " + str(self.v_x) )
        print("y_global: " + str(self.y_global) )
        print("v_y_global: " + str(self.v_y) )        
        print("altitude: " + str(self.alt) )
        print("range_front: " + str(self.dist_front) )
        print("range_right: " + str(self.dist_right) )
        print("range_left: " + str(self.dist_left) )
        print("range_back: " + str(self.dist_back) )
        print("==================================\n")
        
    
        arr = [roll, pitch, yaw_rate, v_x, v_y, self.altitude ,range_front_value, range_back_value ,range_right_value, range_left_value ]
        
        arr = np.array(arr)
        
        #print("DEBUG get_observations "+str(arr))
        
        return arr

    def print_debug_status(self):
    
        print("====== PID input =======\n")
        print("dt   " + str(self.dt) )
        print("height_desired   " + str(self.height_desired) )        
        print("roll  " + str(self.roll) )
        print("Pitch  " + str(self.pitch) )   
        print("Yaw rate: " + str(self.yaw_rate) )              
        print("altitude: " + str(self.alt) )
        print("v_x: " + str(self.v_x) )
        print("v_y: " + str(self.v_y) )
        print("SIMULATION TIME: " + str(self.getTime()) )
        print("==================================\n")
    
        
    def take_off(self):
        """
        Para despegar el drone al inicio.
        Función llamada luego de initialize
        """

        # mientras el drone aún no haya despegado:     
        while self.the_drone_took_off == False:       
            

            if self.alt < MAX_HEIGHT :
                
                if self.height_desired < MAX_HEIGHT: # altura deseada menor al MAXIMO 
                    
                    # altura del drone mayor a la altura deseada
                    if self.alt > self.height_desired:                
                        # incremento altura deseada                            
                        self.height_desired +=  HEIGHT_INCREASE
                                       
            else:
                        
                if self.timestamp_take_off ==0:
                            self.timestamp_take_off = self.getTime()
                        
                if self.getTime() - self.timestamp_take_off > 15:
                    
                    self.the_drone_took_off = True
                    print("DESPEGO " ) 

            # PID velocity controller with fixed height
            motor_power = self.PID_crazyflie.pid(self.dt, 0, 0,
                                            0, self.height_desired ,
                                            self.roll, self.pitch, self.yaw_rate,
                                            self.alt, self.v_x, self.v_y)                                                
            
            self.setup_motors_velocity(motor_power)

            super().step(self.timestep)            
            
            self.past_time = self.getTime()
            self.past_x_global = self.x_global
            self.past_y_global = self.y_global
            # update observations 
            obs = self.get_observations()
            
            #super().step(self.timestep)     
            self.print_debug_status()
        
        
    def step(self, action):
        
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        
        print("DEBUG apply_action "+str(action)) 
        action = int(action)
        
        if action==0:
            forward_desired = 0.5 # go forward
        elif action==1:
            forward_desired = -0.5 # go backwards            
        elif action==2: # move left
            sideways_desired = 0.5
        elif action==3:
            sideways_desired = -0.5 # move right
        elif action==4:
            yaw_desired = 1 # turn left
        elif action==5:
            yaw_desired = -1  # turn right   
        elif action==6: # keep on the same place
            forward_desired = 0
            sideways_desired = 0
            yaw_desired = 0
        #elif action == -22: # drone takes off
        #   print("modo despegue")
        else: 
            print("Not a valid action")               
        
        # PID velocity controller with fixed height
        motor_power = self.PID_crazyflie.pid(self.dt, forward_desired, sideways_desired,
                                        yaw_desired, self.height_desired ,
                                        self.roll, self.pitch, self.yaw_rate,
                                        self.alt, self.v_x, self.v_y)
        self.setup_motors_velocity(motor_power)
        
        
        self.past_time = self.getTime()
        self.past_x_global = self.x_global
        self.past_y_global = self.y_global
            
            
        super().step(self.timestep)
        print("====== PID input ifs =======\n")
        print("forward_desired   " + str(forward_desired) )
        print("sideways_desired   " + str(sideways_desired) )
        print("yaw_desired   " + str(yaw_desired) )
        print("==================================\n")
     
        self.print_debug_status()
     
        # Observation
        obs = self.get_observations()
        
        # Reward
        reward = self.get_reward()

        # observations / state, reward ,done , {}
        return obs, reward, self.is_done(), {}
        
        
    def get_reward(self, action=None):
        
        """
        El objetivo es que el drone aprenda a acercarse 
        a una de las paredes
        
        """
        r = 0 
        

        # muy cerca de la pared, 2000 mm es el máximo, 200 mm = 20 cm  
        #if self.dist_front <= 200 or self.dist_back  <= 200 or self.dist_right <= 200 or self.dist_left  <= 200 :
        #    r = 1
        #    print("DEBUG Reward se está acercando ")
        #else:
        #    r= -1
            

        # penalizo por estar lejos de la pared
        if self.dist_front > 500 and self.dist_back  > 500 and self.dist_right > 500 and self.dist_left  > 500 :
            r += -1
            print("DEBUG Reward muy lejos de la pared ")
        else:
            r += 1

        
        
        # Reward for every step the episode hasn't ended
        print("DEBUG Reward value " + str(r))
    
        return r

    def is_done(self):
        
        if self.episode_score > 195.0:
            return True
        
        # Si ya está el cuadricóptero al menos a 20 cm de la pared
        # doy x terminado el trabajo 
        # 2000 mm es el máximo, 200 mm = 20 cm  
        if bool(self.dist_front <= 200 or
           self.dist_back  <= 200 or 
           self.dist_right <= 200 or 
           self.dist_left  <= 200) :
            return True
              
        
        return False
            

def main():
    # Initialize the environment
    env = DroneOpenAIGymEnvironment()
    #check_env(env)
    
    obs = env.reset()
    # Replay
    # print('Drone took off, press `Y` to continue...')
    # env.wait_keyboard()
    
    # Train
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)
    model.learn(total_timesteps=1e5)
    
    # Save the model
    model.save("ppo_model")

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    
    # lo siguiente no ha llegado a ser ejecutado 
    #print('Drone took off, press `Y` to continue...')
    #env.wait_keyboard()
    
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()

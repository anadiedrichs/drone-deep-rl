"""my_controller controller."""


from controller import Robot
from controller import Keyboard # user input
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor
from math import cos, sin, pi

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import *
#from PPO_agent import PPOAgent, Transition

from gym.spaces import Box
import numpy as np

import sys
sys.path.append('../../../../controllers_shared/python_based')


class DroneRobotSupervisor(RobotSupervisorEnv):

    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        
        #[roll, pitch, yaw_rate, v_x, v_y, self.altitude ,self.range_front_value, self.range_back_value ,self.range_right_value, self.range_left_value ] #4
        
        self.observation_space = Box(low=np.array([- pi, -pi/2, - pi, 0,0, 0.1, 2.0, 2.0, 2.0, 2.0]),
                                     high=np.array([pi, pi/2, pi, 10, 10, 5,2000, 2000, 2000, 2000]),
                                     dtype=np.float64)
        
        # Define agent's action space using Gym's Discrete
        # set the velocity for each motor. The drone has four motors
        self.action_space = Box(low=np.array([-1.0 , -1.0, -1.0, -1.0]),
                                     high=np.array([ 1, 1, 1, 1]),
                                     dtype=np.float64)
            
        #print("Shape of Box space:",self.observation_space.shape)
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        
        timestep = int(self.getBasicTimeStep())
        self.timestep = timestep
        
        # Initialize motors
        self.motors = [None for _ in range(4)]
        self.setup_motors()      
          
        # Initialize Sensors
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/inertialunit.md
        self.imu = self.getDevice("inertial_unit")
        self.imu.enable(timestep)
        self.gps = self.getDevice("gps")
        self.gps.enable(timestep)
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/gyro.md
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(timestep)
        self.camera = self.getDevice("camera") # not used
        self.camera.enable(timestep)
        self.range_front = self.getDevice("range_front")
        self.range_front.enable(timestep)
        self.range_left = self.getDevice("range_left")
        self.range_left.enable(timestep)
        self.range_back = self.getDevice("range_back")
        self.range_back.enable(timestep)
        self.range_right = self.getDevice("range_right")
        self.range_right.enable(timestep)
        
        # Get keyboard, not currently in use
        self.keyboard = Keyboard()
        self.keyboard.enable(timestep)
            
            
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.altitude = 0.0
        self.x_global = 0.0
        self.y_global = 0.0
        self.first_time = True
    
        print("DEBUG init DroneRobotSupervisor")
        

    
    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        
        self.motors[0] = self.getDevice('m1_motor')
        self.motors[1] = self.getDevice('m2_motor')
        self.motors[2] = self.getDevice('m3_motor')
        self.motors[3] = self.getDevice('m4_motor')
        
        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))
            if i % 2 != 0 :
                self.motors[i].setVelocity(-1.0) # motor 1 and 3
            else:
                self.motors[i].setVelocity(1.0) # motor 2 and 4
        

            
    def get_observations(self):
    
        self.dt = self.getTime() - self.past_time

        if self.first_time:
            self.past_x_global = self.gps.getValues()[0]
            self.past_y_global = self.gps.getValues()[1]
            self.past_time = self.getTime()
            self.first_time = False

        # Get sensor data
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/inertialunit.md
        roll = self.imu.getRollPitchYaw()[0]
        pitch = self.imu.getRollPitchYaw()[1]
        yaw = self.imu.getRollPitchYaw()[2]
        yaw_rate = self.gyro.getValues()[2] # The angular velocity is measured in radians per second [rad/s].
        self.x_global = self.gps.getValues()[0]
        v_x_global = (self.x_global - self.past_x_global)/self.dt
        self.y_global = self.gps.getValues()[1]
        v_y_global = (self.y_global - self.past_y_global)/self.dt
        
        self.alt = self.gps.getValues()[2]
        self.altitude = normalize_to_range(self.alt ,0.1, 5, -1.0, 1.0,clip=True)

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # camera_data = camera.getImage()

        # to get the value in meters you have to divide by 1000
        self.dist_front = self.range_front.getValue()
        self.range_front_value = normalize_to_range(self.dist_front,2, 2000, -1.0, 1.0,clip=True)
        self.dist_back = self.range_back.getValue()
        self.range_back_value = normalize_to_range(self.dist_back,2, 2000, -1.0, 1.0,clip=True)
        self.dist_right = self.range_right.getValue()
        self.range_right_value = normalize_to_range(self.dist_right,2, 2000, -1.0, 1.0,clip=True)
        self.dist_left = self.range_left.getValue()
        self.range_left_value = normalize_to_range(self.dist_left,2, 2000, -1.0, 1.0,clip=True)
    
        # TODO Normalizar roll, pitch yaw vx vy
        roll = normalize_to_range(roll,- pi, pi, -1.0, 1.0,clip=True)
        pitch = normalize_to_range(pitch,- pi/2, pi/2, -1.0, 1.0,clip=True)
        yaw_rate = normalize_to_range(yaw_rate,- pi, pi, -1.0, 1.0,clip=True)
        v_x = normalize_to_range(v_x,0, 10, -1.0, 1.0,clip=True)
        v_y = normalize_to_range(v_y,0, 10, -1.0, 1.0,clip=True)
        
    
        arr = [roll, pitch, yaw_rate, v_x, v_y, self.altitude ,self.range_front_value, self.range_back_value ,self.range_right_value, self.range_left_value ] #4
        
        arr = np.array(arr)
        
        print("DEBUG get_observations "+str(arr))
        
        return arr


    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        
        r = 0 
        
        # demasiado cerca de la pared 
        #if self.range_front_value < 0.1 or 
        #self.range_back_value  < 0.1 or
        #self.range_right_value  < 0.1 or
        #self.range_left_value  < 0.1 :
        #    r += -1
        
        # muy cerca de la pared 
        if self.dist_front < 200 or self.dist_back  < 200 or self.dist_right < 200 or self.dist_left  < 200 :
            r += -10
            print("DEBUG Reward se está acercando mucho ")
        

        # muy lejos de la pared
        if self.dist_front > 200 and self.dist_back  > 200 and self.dist_right > 200 and self.dist_left  > 200 :
            r += -10
            print("DEBUG Reward muy lejos de la pared ")
        else:
            r += 10
        # check the drone's altitude     
        if self.alt < 1 or self.alt > 1.5 :
            r += - 10
            print("DEBUG Reward altura inadecuada " + str(r))
        else: 
            r += 10
        
        
        # Reward for every step the episode hasn't ended
        print("DEBUG Reward value " + str(r))
    
        return r

    def is_done(self):
        
        if self.episode_score > 195.0:
            return True

        #pole_angle = round(self.position_sensor.getValue(), 2)
        #if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
        #    return True

        #cart_position = round(self.robot.getPosition()[0], 2)  # Position on x-axis
        #if abs(cart_position) > 0.39:
        #    return True

        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                print("DEBUG solved True ")
                return True
                
        print("DEBUG solved False")
        return False

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'): 
        pass

    def take_off(self):
        print("DEBUG take_off ") 
        for i in range(len(self.motors)):
        
            if i % 2 != 0 :
                self.motors[i].setVelocity(-10.0) # motor 1 and 3
            else:
                self.motors[i].setVelocity(10.0) # motor 2 and 4
        
    def apply_action(self, action):
    
        
        print("DEBUG apply_action "+str(action)) 
        
        # the first action is to take off (despegar )
        if self.first_time:
            print("DEBUG apply_action PRIMERA VEZ QUE ENTRO ") 
            self.take_off()
            
        else: 
        
            if self.getTime() < 10: 
                self.take_off()
            else:
            
                for i in range(len(self.motors)):
                    aux =  convert_to_interval_0_600(float(action[i]))
                    print("DEBUG action %d : %f" % (i,aux))
                    if i % 2 != 0 :
                       self.motors[i].setVelocity(-aux) # motor 1 and 3
                    else:
                       self.motors[i].setVelocity(aux) # motor 2 and 4
                    
        self.past_time = self.getTime()
        self.past_x_global = self.x_global
        self.past_y_global = self.y_global
        
        
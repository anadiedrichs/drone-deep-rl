"""
More runners for discrete RL algorithms can be added here.
"""
import sys
import pandas as pd
from controller import Supervisor
from controller import Keyboard  # user input
from math import cos, sin, pi
from gymnasium.spaces import Sequence, Box, Discrete, Tuple
from utils.utilities import *
from utils.pid_controller import *
from pilots.pilot import *

MAX_HEIGHT = 0.7  # in meters
HEIGHT_INCREASE = 0.05  # 5 cm
HEIGHT_INITIAL = 0.30  # 20 cm
WAITING_TIME = 5  # in seconds
# 2000 mm es el máximo, 200 mm = 20 cm
# DIST_MIN = 1000  # in mm (200 + 200 + 1800 + 1800 )/4

try:
    import gymnasium as gym  #nasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gymnasium stable_baselines3"'
    )


class DroneRobotSupervisor(Supervisor, gym.Env):
    """
    Implementation of the Crazyflie environment for Webots.
    """

    def __init__(self, max_episode_steps=30_000):
        super().__init__()
        self.render_mode = None  # no render need: None
        self.max_episode_steps = max_episode_steps
        # a vector (vx,vy,vz)
        self.lineal_velocity = None
        # a vector (omega_x,omega_y,omega_z)
        self.angular_velocity = None
        self.dist_left = 0
        self.dist_right = 0
        self.dist_back = 0
        self.past_y_global = 0
        self.past_x_global = 0
        self.z_global = 0
        self.past_z_global = 0
        self.dist_front = 0
        self.pilot_action = None
        self.obs_copilot = None  # copilot input
        self.training_mode = False  # True if we are training a copilot
        self.obs_array = None  # base-pilot input
        self.observation_space = self._get_observation_space()

        # Define agent's action space using Gym's Discrete
        self.action_space = Discrete(6)

        self.spec = gym.envs.registration.EnvSpec(id='CrazyflieWebotsEnv-v0', max_episode_steps=max_episode_steps)

        # 2) Environment specific configuration
        if self.getBasicTimeStep() != 32:
            raise ValueError("WorldInfo.basicTimeStep should be 32 for this project.")
        else:
            self.timestep = int(self.getBasicTimeStep())
        # Get keyboard, used by trajectory generation
        self._keyboard = Keyboard()
        self._keyboard.enable(self.timestep)
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
        self.episode_step = 0
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
        self.v_z = 0
        self.alt = 0  # in meters
        self.past_time = 0
        self.the_drone_took_off = False
        self.height_desired = HEIGHT_INITIAL
        self.timestamp_take_off = 0
        self.timer1 = 0
        # Initialize motors
        self.motors = None
        self.is_success = False
        self.terminated = False
        self.truncated = False
        self.pilot = None
        self.alpha = 0.5  # copilot coefficient
        # to be done ?
        # self.screenshoot_counter = 0
        self.pen_activated = False
        # crazyflie reference
        self.robot = self.getFromDef('crazyflie')
        self.drone_trajectory_file = None
        self.drone_trajectory_path = None
        self.x_pos_initial = None
        self.y_pos_initial = None

        self._initialization()

        print("DEBUG init DroneRobotSupervisor")


    def _initialization(self):
        """
        Internal function for initial sensors and actuators setup.
        It must be called always after __init__ and reset.
        """
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
        self.camera = self.getDevice("camera")  # not used
        self.camera.enable(self.timestep)
        self.range_front = self.getDevice("range_front")
        self.range_front.enable(self.timestep)
        self.range_left = self.getDevice("range_left")
        self.range_left.enable(self.timestep)
        self.range_back = self.getDevice("range_back")
        self.range_back.enable(self.timestep)
        self.range_right = self.getDevice("range_right")
        self.range_right.enable(self.timestep)
        # a vector (vx,vy,vz)
        self.lineal_velocity = None
        # a vector (omega_x,omega_y,omega_z)
        self.angular_velocity = None
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.episode_step = 0
        self.x_global = 0.0
        self.y_global = 0.0
        self.z_global = 0.0
        self.first_time = True
        self.dt = self.timestep
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.yaw_rate = 0
        self.v_x = 0
        self.v_y = 0
        self.v_z = 0
        self.alt = 0  # in meters
        self.lineal_velocity = None
        self.angular_velocity = None
        self.past_time = 0
        self.the_drone_took_off = False
        self.height_desired = HEIGHT_INITIAL
        self.timestamp_take_off = 0
        self.timer1 = 0
        self.is_success = False
        self.terminated = False
        self.truncated = False
        # Initialize motors
        self.motors = [None for _ in range(4)]
        self._setup_motors()
        # base-pilot & copilot
        # set base-pilot via set_pilot method please
        # self.base-pilot = None
        self.alpha = 0.5  # copilot coefficient
        self.obs_array = self.get_default_observation()
        self.obs_copilot = self.get_default_observation()
        # self.screenshoot_counter = 0
        self.pen_activated = False
        self.drone_trajectory_file = None
        self.x_pos_initial = None
        self.y_pos_initial = None
        # Crazyflie reference
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError(
                "El nodo Crazyflie no fue encontrado. Verifica el identificador DEF en el archivo del mundo.")
        # else:
        #     print("Nodo Crazyflie encontrado.")
        #
        # # Inicializar el dispositivo Pen sin activarlo todavía
        # self._init_pen()

        print("DEBUG initialization")

    def increment_episode_step(self):
        self.episode_step += 1


    def get_trajectory_file_name(self):
        if self.drone_trajectory_file is None:
            ValueError("trajectory file name must be set up. Use set_trajectory_path ")

        return self.drone_trajectory_file
    def set_trajectory_path(self, path_dir):
        if self.drone_trajectory_path is None:
            self.drone_trajectory_path = path_dir

    def set_trajectory_file_name(self, file_name):
        if self.drone_trajectory_file is None:
            if self.drone_trajectory_path is None:
                ValueError("Set first the drone_trajectory_path by calling set_trajectory_path")

            self.drone_trajectory_file = self.drone_trajectory_path + file_name
            # Inicializar el archivo CSV para guardar la trayectoria en modo append
            with open(self.drone_trajectory_file, "w") as f:
                f.write("time,x,y,z\n")  # Escribir encabezado del archivo

    def record_trajectory(self):
        # Obtener la posición actual del drone
        position = self.gps.getValues()
        current_time = self.getTime()
        # Agregar el tiempo y la posición al archivo CSV
        with open(self.get_trajectory_file_name(), "a") as f:
            f.write(f"{current_time},{position[0]},{position[1]},{position[2]}\n")

    def _init_pen(self):
        """
        Internal function to initialize and attach a Pen to the Crazyflie robot.
        """
        pen_string = """
            Pen {
              name "pen"
            }
            """

        # Obtener el campo extensionSlot del Crazyflie
        extension_slot_field = self.robot.getField("extensionSlot")
        if extension_slot_field is None:
            raise ValueError("El campo extensionSlot no fue encontrado en el nodo Crazyflie.")

        # Agregar el nodo Pen al extensionSlot
        extension_slot_field.importMFNodeFromString(-1, pen_string)

        print("DEBUG: Pen añadido al extensionSlot del Crazyflie")

    def _setup_motors(self):
        """
        This method initializes the four Crazyflie propellers / motors.
        """

        self.motors[0] = self.getDevice('m1_motor')
        self.motors[1] = self.getDevice('m2_motor')
        self.motors[2] = self.getDevice('m3_motor')
        self.motors[3] = self.getDevice('m4_motor')

        for i in range(len(self.motors)):
            self.motors[i].setPosition(float('inf'))

        self._setup_motors_velocity(np.array([1.0, 1.0, 1.0, 1.0]))

    def _get_observation_space(self) -> Box:

        # 1) OpenAIGym generics
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values

        # [roll, pitch, yaw_rate, v_x, v_y, self.altitude ,self.range_front_value, self.range_back_value ,self.range_right_value, self.range_left_value ] #4

        #self.observation_space = Box(low=np.array([- pi, -pi / 2, - pi, 0, 0, 0.1, 2.0, 2.0, 2.0, 2.0]),
        #                             high=np.array([pi, pi / 2, pi, 10, 10, 5, 2000, 2000, 2000, 2000]),
        #
        return Box(low=-1, high=1, shape=(10,), dtype=np.float64)

    def _set_observation_space(self, b: Box):
        """

        * Observation space: 10 continuous variables.
        [roll, pitch, yaw_rate, v_x, v_y, altitude,
        range_front_value,range_back_value,
        self.range_right_value, self.range_left_value ]

        * Action space: 7 discrete actions.
        See step() method implementation.
        """
        self.observation_space = b

    def _setup_motors_velocity(self, motor_power):
        """
        Sets the velocity for each motor in the drone.

        This method takes a list of motor power values and applies them to the
        corresponding motors. Negative values are applied to some motors to account
        for their orientation or intended direction of rotation.

        Parameters:
        motor_power (list of float): A list of four float values representing the
        power to be applied to each motor. The order corresponds to the motors'
        positions:
            - motor_power[0]: Power for the first motor (inverted)
            - motor_power[1]: Power for the second motor
            - motor_power[2]: Power for the third motor (inverted)
            - motor_power[3]: Power for the fourth motor

        The motors are assumed to be positioned such that the first and third motors
        require inversion (negative power) for the drone to move in the intended direction.
        """

        self.motors[0].setVelocity(-motor_power[0])
        self.motors[1].setVelocity(motor_power[1])
        self.motors[2].setVelocity(-motor_power[2])
        self.motors[3].setVelocity(motor_power[3])
        # print("====== Motors velocity =======\n")
        # print(" m1 " + str(-motor_power[0]))  # 1
        # print(" m2 " + str(motor_power[1]))  # 2
        # print(" m3 " + str(-motor_power[2]))  # 3
        # print(" m4 " + str(motor_power[3]))  # 4

    def render(self, mode="human"):
        """render method is not used"""
        pass

    def wait_keyboard(self):
        while self._keyboard.getKey() != ord('Y'):
            super().step(self.timestep)

    def get_webots_keyboard(self):
        return self._keyboard

    def get_default_observation(self):
        """
         This method just returns a zero vector as a default observation
        """
        return np.array([0.0 for _ in range(self.observation_space.shape[0])])

    def reset(self, seed=None, options=None):
        """
        Reset the simulation
        """
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.timestep)

        # reset to initial values
        self._initialization()

        # Internals
        super().step(self.timestep)

        self.take_off()

        file_name = "trajectory_x_"+str(self.x_global)+"_y_"+str(self.y_global)
        self.set_trajectory_file_name(file_name)
        self.x_pos_initial = self.x_global
        self.y_pos_initial = self.y_global

        # # Activar el Pen después de que el drone haya despegado
        # extension_slot_field = self.robot.getField("extensionSlot")
        # pen_node = extension_slot_field.getMFNode(-1)
        #
        # # Verificar si el nodo Pen fue agregado correctamente
        # if pen_node is None:
        #     raise ValueError("El nodo Pen no fue encontrado después de agregarlo.")
        #
        # # Activar el Pen para dibujar
        # pen_write_field = pen_node.getField("write")
        # if pen_write_field is None:
        #     raise ValueError("El campo 'write' no fue encontrado en el nodo Pen.")
        # pen_write_field.setSFBool(True)
        #
        # # Establecer el color del Pen
        # pen_ink_color_field = pen_node.getField("inkColor")
        # if pen_ink_color_field is None:
        #     raise ValueError("El campo 'inkColor' no fue encontrado en el nodo Pen.")
        # pen_ink_color_field.setSFColor([0.0, 0.0, 1.0])  # Color azul
        #
        # print("DEBUG: Pen activado después del despegue")

        # Open AI Gym generic
        return self.get_default_observation(), {}  # empty info dict

    def transform_velocity(self, v_x_global, v_y_global):
        # Calcular coseno y seno de yaw
        cos_yaw = cos(self.yaw)
        sin_yaw = sin(self.yaw)

        # Transformar velocidades globales a locales
        self.v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        self.v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        return self.v_x, self.v_y
    def update_lineal_velocity(self):
        # Get position
        self.x_global = self.gps.getValues()[0]
        self.y_global = self.gps.getValues()[1]
        self.z_global = self.gps.getValues()[2]
        self.alt = self.gps.getValues()[2]
        # Calculate global velocity en X, Y y Z
        v_x_global = (self.x_global - self.past_x_global) / self.dt
        v_y_global = (self.y_global - self.past_y_global) / self.dt
        v_z_global = (self.z_global - self.past_z_global) / self.dt
        # Almacenar la posición global actual como pasada para el próximo cálculo
        self.past_x_global = self.x_global
        self.past_y_global = self.y_global
        self.past_z_global = self.z_global
        # Get body fixed velocities
        self.transform_velocity(v_x_global, v_y_global)
        self.v_z = v_z_global
        # camera_data = camera.getImage()
        self.lineal_velocity = np.array([self.v_x,self.v_y,self.v_z])

    def get_observations(self):
        """
        Read sensors values
        """

        if self.first_time:
            self.past_x_global = self.gps.getValues()[0]
            self.past_y_global = self.gps.getValues()[1]
            self.past_z_global = self.gps.getValues()[2]
            self.past_time = self.getTime()
            # a vector (omega_x,omega_y,omega_z)
            self.angular_velocity = np.array(self.gyro.getValues())
            self.first_time = False

        self.dt = self.getTime() - self.past_time

        # print("dt   " + str(self.dt) )
        # print("past_time   " + str(self.past_time) )
        # print("getTime   " + str(self.getTime()) )
        if self.dt == 0:  # solve division by zero bug
            self.dt = self.timestep

        # Get sensor data
        # https://github.com/cyberbotics/webots-doc/blob/master/reference/inertialunit.md
        self.roll = self.imu.getRollPitchYaw()[0]
        self.pitch = self.imu.getRollPitchYaw()[1]
        self.yaw = self.imu.getRollPitchYaw()[2]
        # The angular velocity is measured in radians per second [rad/s].c
        self.yaw_rate = self.gyro.getValues()[2]
        self.angular_velocity = np.array(self.gyro.getValues())
        # update position and velocity
        self.update_lineal_velocity()
        # to get the value in meters you have to divide by 1000
        self.dist_front = self.range_front.getValue()
        self.dist_back = self.range_back.getValue()
        self.dist_right = self.range_right.getValue()
        self.dist_left = self.range_left.getValue()

        roll = normalize_to_range(self.roll, - pi, pi, -1.0, 1.0, clip=True)
        pitch = normalize_to_range(self.pitch, - pi / 2, pi / 2, -1.0, 1.0, clip=True)
        yaw_rate = normalize_to_range(self.yaw_rate, - pi, pi, -1.0, 1.0, clip=True)
        v_x = normalize_to_range(self.v_x, 0, 10, -1.0, 1.0, clip=True)
        v_y = normalize_to_range(self.v_y, 0, 10, -1.0, 1.0, clip=True)
        altitude = normalize_to_range(self.alt, 0.1, 5, -1.0, 1.0, clip=True)
        range_front_value = normalize_to_range(self.dist_front, 2, 2000, -1.0, 1.0, clip=True)
        range_back_value = normalize_to_range(self.dist_back, 2, 2000, -1.0, 1.0, clip=True)
        range_right_value = normalize_to_range(self.dist_right, 2, 2000, -1.0, 1.0, clip=True)
        range_left_value = normalize_to_range(self.dist_left, 2, 2000, -1.0, 1.0, clip=True)

        # self.print_debug_status()

        arr = [roll, pitch, yaw_rate,
               v_x, v_y,
               altitude, range_front_value, range_back_value, range_right_value,
               range_left_value]

        arr = np.array(arr)
        # we store in obs_array only the sensors observations (10 values)
        self.obs_array = arr
        # for copilot input (11 values)

        #print("DEBUG get_observations "+str(arr))
        if self.pilot is not None:
            action, _ = self.pilot.choose_action(self.obs_array)
            arr = np.append(arr, normalize_to_range(action, 0, 6, -1.0, 1.0, clip=True))
            self.obs_copilot = np.array(arr)

        return arr

    def print_debug_status(self):
        """
        Print sensors values
        """

        print("====== PID input =======\n")
        print("dt   " + str(self.dt))
        print("altitude: " + str(self.alt))
        print("height_desired   " + str(self.height_desired))
        print("roll  " + str(self.roll))
        print("Pitch  " + str(self.pitch))
        print("Yaw rate: " + str(self.yaw_rate))
        print("v_x: " + str(self.v_x))
        print("v_y: " + str(self.v_y))
        # print("==================================\n")
        # print("SIMULATION TIME: " + str(self.getTime()) )
        # print("====== SENSORS observations =======\n")
        print("Yaw    " + str(self.yaw))
        print("Yaw rate: " + str(self.yaw_rate))
        print("x_global: " + str(self.x_global))
        print("v_x_global: " + str(self.v_x))
        print("y_global: " + str(self.y_global))
        print("v_y_global: " + str(self.v_y))
        print("range_front: " + str(self.dist_front))
        print("range_right: " + str(self.dist_right))
        print("range_left: " + str(self.dist_left))
        print("range_back: " + str(self.dist_back))
        print("==================================\n")

    def take_off(self):

        """
        This function must be called at the beginning of the simulation
        to take off the drone.
        """

        while not self.the_drone_took_off:

            if self.alt < MAX_HEIGHT:

                if self.alt > self.height_desired:
                    # incremento altura deseada
                    self.height_desired = self.height_desired + HEIGHT_INCREASE
                    # reset timer pues aun no se estabiliza
                    self.timestamp_take_off = 0
            else:
                # inicializo timer de espera post despegue
                if self.timestamp_take_off == 0:
                    self.timestamp_take_off = self.getTime()

                if self.getTime() - self.timestamp_take_off > WAITING_TIME:
                    self.timestamp_take_off = 0
                    self.the_drone_took_off = True
                    self.height_desired = self.alt
                    print("CRAZYFLIE TOOK OFF ! ")

            motor_power = self.PID_crazyflie.pid(self.dt, 0, 0,
                                                 0, self.height_desired,
                                                 self.roll, self.pitch, self.yaw_rate,
                                                 self.alt, self.v_x, self.v_y)
            self._setup_motors_velocity(motor_power)
            super().step(self.timestep)

            # update sensor readings
            obs = self.get_observations()

            #self.print_debug_status()

            self.past_time = self.getTime()
            self.past_x_global = self.x_global
            self.past_y_global = self.y_global
            self.past_z_global = self.z_global

    def step(self, action):

        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        self.increment_episode_step()
        action = int(action)

        if self.pilot is not None:
            self.pilot_action, _ = self.pilot.choose_action(self.obs_array)
            if self.training_mode:
                print("copilot training ")
                action = self.pilot_action
            else:
                print("copilot testing ")
                # add alpha_parameter
                action, _ = self.choose_action(self.obs_copilot)
        print("ACTION number " + str(action))
        if action == 0:
            forward_desired = 0.005  # go forward
        elif action == 1:
            forward_desired = -0.005  # go backwards
        elif action == 2:  # move left
            sideways_desired = 0.005
        elif action == 3:
            sideways_desired = -0.005  # move right
        elif action == 4:
            yaw_desired = 0.01  # turn left
        elif action == 5:
            yaw_desired = -0.01  # turn right
        # elif action == 6:  # keep on the same place
        #    forward_desired = 0
        #    sideways_desired = 0
        #    yaw_desired = 0
        else:
            print("Not a valid action")

            # PID velocity controller with fixed height
        motor_power = self.PID_crazyflie.pid(self.dt, forward_desired, sideways_desired,
                                             yaw_desired, self.height_desired,
                                             self.roll, self.pitch, self.yaw_rate,
                                             self.alt, self.v_x, self.v_y)
        self._setup_motors_velocity(motor_power)

        super().step(self.timestep)
        # print("====== PID input ACTION =======\n")
        # print("height_desired   " + str(self.height_desired) )
        # print("forward_desired   " + str(forward_desired) )
        # print("sideways_desired   " + str(sideways_desired) )
        # print("yaw_desired   " + str(yaw_desired) )
        # print("==================================\n")
        # self.print_debug_status()
        # get the environment observation (sensor readings)
        obs = self.get_observations()
        # Reward
        reward = self.get_reward(action)
        # update times
        self.past_time = self.getTime()
        #self.past_x_global = self.x_global
        #self.past_y_global = self.y_global
        # successful episode ?
        done = self.is_done()
        info = self.get_info()
        # Registrar la trayectoria en cada paso
        self.record_trajectory()
        return obs, reward, done, self.truncated, info

    def set_pilot(self, p: Pilot):
        self.pilot = p

    def get_info(self):
        """
            info = {"is_success": self.is_success}
        """
        raise NotImplementedError("Please Implement this method")

    def get_reward(self, action=6):
        """
        """
        raise NotImplementedError("Please Implement this method")
    def is_done(self):
        """
        Return True when a final state is reached.
        """
        raise NotImplementedError("Please Implement this method")

    def get_info_keywords(self):
        """
        Return a Tuple[str, ...] to set info_keywords in Monitor
        Check https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/monitor.html#Monitor
        """

        raise NotImplementedError("Please Implement this method")

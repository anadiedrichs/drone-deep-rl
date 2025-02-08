import numpy as np
import math
from math import cos, sin, sqrt
from copilot.CrazyflieDrone import *
from copilot.rotation import *
from copilot.location import *

class CornerEnv(DroneRobotSupervisor):
    """
    A Crazyflie base-pilot with the mission of reaching a target
    in a square room.
    """
    CORNERS = ["cone_1", "cone_2", "cone_3", "cone_4"]
    TARGET_LIST = []
    # 10 cm
    DISTANCE_THRESHOLD = 0.15
    # 100 mm
    MIN_DIST_OBSTACLES = 100
    FILE_NAME_ROTATION = "rotation.txt"
    FILE_NAME_LOCATION = "location.txt"
    MIN_EPISODE_SCORE = -50_000
    MAX_EPISODE_SCORE = 50_000

    def __init__(self, max_episode_steps=30_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.corner = None
        self.x_init = None
        self.y_init = None
        self.closest_corner = None
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")

        self._init_targets()
        self._initialization()

    def _init_targets(self):
        # 1) get TARGET info
        self.corner = None
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        for name in self.CORNERS:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)

    def _init_rotation(self):
        # 2) ROTATION CHANGE
        # Leer la rotación desde el archivo (que ahora está actualizada)
        rotacion = leer_rotacion_desde_archivo(self.FILE_NAME_ROTATION)
        #if rotacion:
        #    print(f"Rotación leída: {rotacion}")
        # Asignar la rotación al drone
        self.robot.getField("rotation").setSFRotation(rotacion)
        # Incrementar la rotación en el eje Z y actualizar el archivo
        incrementar_rotacion_en_z(self.FILE_NAME_ROTATION)

    def _init_location(self):
        # read the drone init location from file
        ubicacion_leida = leer_ubicacion_desde_archivo(self.FILE_NAME_LOCATION)
        if ubicacion_leida:
            #print(f"Ubicación leída: {ubicacion_leida}")
            # Establecer la nueva ubicación (modificando sólo las coordenadas x e y)
            position_field = self.robot.getField("translation")
            z = position_field.getSFVec3f()[2]  # Mantener la coordenada Z actual
            nueva_posicion = [ubicacion_leida[0], ubicacion_leida[1], z]
            self.x_init = ubicacion_leida[0]
            self.y_init = ubicacion_leida[1]
            position_field.setSFVec3f(nueva_posicion)

        # 3) CHANGE drone location in (x,y)
        ubicacion = generar_ubicacion_aleatoria()
        #print(f"Ubicación aleatoria generada: {ubicacion}")
        # Guardar la nueva ubicación en el archivo
        guardar_ubicacion_en_archivo(self.FILE_NAME_LOCATION, ubicacion)

    def _initialization(self):
        super()._initialization()
        print("DEBUG episode score "+str(self.episode_score))
        self.dist_min_target = 0
        self.corner = None
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self._init_location()
        self._init_rotation()

    def get_distance_to_corners(self):
        d = []
        for t in self.TARGET_LIST:
            d.append(get_distance_from_target(self.robot, t))
        #print(d)
        d = np.array(d)
        return d

    def is_done(self):
        """
        Method call by step function to know if the episode ends.
        Return True when:
         * the drone reach a target corner
         * the drone falls
         * reward reaches a threshold
         """
        done = False
        # 1) if the drone fell to the ground
        print("self.alt ", str(self.alt))
        # if the drone falls, the episode is truncated
        if self.alt < 0.1:  # alt is in meters
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "fall"
        # 2) if the drone reach a corner cone
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.truncated = False
            self.corner = self.CORNERS[np.argmin(distance)]
        # 3) it accumulates too much score without reaching the corner
        if (self.episode_score <= self.MIN_EPISODE_SCORE or
            self.episode_score >= self.MAX_EPISODE_SCORE):
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "score"
        if self.episode_step > self.max_episode_steps:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "max_steps"
        # Webots bug, webots freezes
        # if done:
        #     # Take a picture of the scene
        #     # Nombre del archivo y calidad (0-100)
        #     self.exportImage(str(self.screenshoot_counter)+"-captura_pantalla.png", 100)
        #     self.screenshoot_counter += 1
        return done

    def get_info_keywords(self):
        return tuple(["x_init", "y_init", "closest_corner", "is_success",
                      "corner", "height", "dist_min_target"])

    def get_info(self):
        """
        Method call by step function
        """
        distance = self.get_distance_to_corners()
        self.closest_corner = self.CORNERS[np.argmin(distance)]
        return {"x_init": self.x_init,
                "y_init": self.y_init,
                "closest_corner": self.closest_corner,
                "is_success": self.is_success,
                "corner": self.corner,
                "height": self.alt,
                "dist_min_target": self.dist_min_target}

    def normalize(self, vector):
        magnitude = sqrt(sum(comp ** 2 for comp in vector))
        return [comp / magnitude for comp in vector]

    def dot_product(self, v1, v2):
        return sum(comp1 * comp2 for comp1, comp2 in zip(v1, v2))

    def get_direction_alignment(self, robot_node, target_node):
        # Obtén la posición y velocidad actuales del robot
        robot_position = robot_node.getField('translation').getSFVec3f()
        robot_velocity = robot_node.getVelocity()[:3]  # Solo componentes de velocidad lineal (x, y, z)

        # Obtén la posición del objetivo
        target_position = target_node.getField('translation').getSFVec3f()

        # Calcula el vector de dirección al objetivo
        direction_to_target = [target_position[i] - robot_position[i] for i in range(3)]

        # Normaliza los vectores
        normalized_velocity = self.normalize(robot_velocity)
        normalized_direction_to_target = self.normalize(direction_to_target)

        # Calcula el producto punto
        alignment = self.dot_product(normalized_velocity, normalized_direction_to_target)
        print("alignment   " + str(alignment))
        return alignment

    def walls_proximity_penalization(self, x):
        #return -10 * np.exp(-(x/0.2))
        # return -10 * np.exp(-(x / 0.4))
        return -7 * np.exp(-(x / 0.1))

    def penalization_distance_to_target(self, x, alpha=5, beta=1):
        #print(f"Tipo de x: {type(x)}, Valor de x: {x}")
        return np.exp(-alpha * x) - (beta * x)

    def episode_steps_penalization(self, x):
        return -5 * np.log(1 + x)

    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # Obtener la distancia mínima a las esquinas
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("DEBUG  dist_min " + str(self.dist_min_target))
        # 1) penalize if the drone is not near a target by its distance
        # pen_dist= self.penalization_distance_to_target(self.dist_min_target)
        # print("DEBUG  pen_dist " + str(pen_dist))
        # get the closest target
        # targ = self.TARGET_LIST[np.argmin(distance)]
        # pen_dir= self.get_direction_alignment(self.robot,targ)
        #print("DEBUG  pen_dist " + str(pen_dir))
        #reward = (self.prev_dist_min_target - pen_dist) + (self.prev_dir_target-pen_dir)
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        #print(obs_min / 100)
        #reward += self.walls_proximity_penalization(obs_min/100)
        # 3) increase the penalty as the number of steps grows
        # reward += self.episode_steps_penalization(self.episode_step)
        # 1) drone reaches the target
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 10
        # 2) penalize the drone if it is too close to an obstacle
        if obs_min <= self.MIN_DIST_OBSTACLES:
            reward = -5
        # 3) penalize the drone if it is too close to an obstacle
        #if (self.episode_score <= self.MIN_EPISODE_SCORE or
        #            self.episode_score >= self.MAX_EPISODE_SCORE):
        #    reward = -10
        # 4) if the drone falls, the episode is truncated
        if self.alt < 0.1:
            reward = -10
        # 5) otherwise
        if reward == 0:
            reward = 1800 * (self.prev_dist_min_target - self.dist_min_target)

        self.prev_dist_min_target = self.dist_min_target
        # self.prev_dir_target = pen_dir

        print("Reward value " + str(reward))
        # update cummulative reward
        self.episode_score += reward
        print("Episode score " + str(self.episode_score))
        # Calcular valores mínimo y máximo de recompensa
        # Depende de las dimensiones del escenario
        # min_reward = self.episode_steps_penalization(self.max_episode_steps)  #
        #min_reward = -61.54
        #max_reward = 15  #
        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, -20, 20, -1, 1)
        # get a reward between -1 and 1
        #normalized_reward = np.tanh(reward)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

    def get_reward_256(self, action=6):
        """
         Method call by step function.
         Recompensa inversamente proporcional a la distancia
         al cono más cercano (conos ubicados en la esquina)
        """
        reward = 0
        distance = self.get_distance_to_corners()
        print("DEBUG  distance " + str(distance))
        self.dist_min_target = distance.min()
        print("DEBUG  dist_min " + str(self.dist_min_target))
        # 1) Recompensa inversamente proporcional a la distancia a la esquina
        # Factor multiplicativo ajustable ? ejemplo * 10
        reward = 10 * (1 / (self.dist_min_target + 0.01))
        print("DEBUG  reward " + str(reward))
        # 4) big penalty if it takes to long to reach a corner
        if self.episode_score <= -100_000 and self.dist_min_target > self.DISTANCE_THRESHOLD:
            reward = -10
        print("Reward value " + str(reward))
        # update cummulative reward
        self.episode_score += reward
        print("Episode score " + str(self.episode_score))
        # Calcular valores mínimo y máximo de recompensa
        # Depende de las dimensiones del escenario
        min_reward = -10  #
        max_reward = 9.09  #
        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, min_reward, max_reward, -1, 1)
        # print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

    def is_in_the_corner(self, min_distance=500):
        """
            Return True if the distance sensors detect a corner at min_distance value
        """
        assert min_distance is not None
        assert min_distance < 2000
        assert min_distance > 10

        return bool((self.dist_front <= min_distance and self.dist_left <= min_distance) or \
                    (self.dist_front <= min_distance and self.dist_right <= min_distance) or \
                    (self.dist_back <= min_distance and self.dist_left <= min_distance) or \
                    (self.dist_back <= min_distance and self.dist_right <= min_distance))


class SimpleCornerEnv_w(DroneRobotSupervisor):
    """
    A Crazyflie base-pilot with the mission of reaching a target
    in a square room.
    """
    CORNERS = ["cone_1"]
    TARGET_LIST = []
    # 10 cm
    DISTANCE_THRESHOLD = 0.15
    # 100 mm
    MIN_DIST_OBSTACLES = 100
    FILE_NAME_ROTATION = "rotation.txt"
    FILE_NAME_LOCATION = "location.txt"
    MIN_EPISODE_SCORE = -50_000
    MAX_EPISODE_SCORE = 50_000

    def __init__(self, max_episode_steps=7_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.corner = None
        self.x_init = None
        self.y_init = None
        self.status = None
        self.closest_corner = None
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")
        self.past_lineal_velocity = None
        self.past_angular_velocity = None
        self._init_targets()
        self._initialization()

    def _init_targets(self):
        # 1) get TARGET info
        self.corner = None
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        for name in self.CORNERS:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)

    def put_drone_in_the_center(self):
        position_field = self.robot.getField("translation")
        z = position_field.getSFVec3f()[2]  # Mantener la coordenada Z actual
        nueva_posicion = [0, 0, z]
        self.x_init = nueva_posicion[0]
        self.y_init = nueva_posicion[1]
        position_field.setSFVec3f(nueva_posicion)

    def _initialization(self):
        super()._initialization()
        print("DEBUG episode score "+str(self.episode_score))
        self.dist_min_target = 0
        self.corner = None
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.status = None
        self.past_lineal_velocity = None
        self.past_angular_velocity = None
        self.put_drone_in_the_center()
        #self._init_rotation()

    def get_distance_to_corners(self):
        d = []
        for t in self.TARGET_LIST:
            d.append(get_distance_from_target(self.robot, t))
        #print(d)
        d = np.array(d)
        return d

    def is_done(self):
        """
        Method call by step function to know if the episode ends.
        Return True when:
         * the drone reach a target corner
         * the drone falls
         * reward reaches a threshold
         """
        done = False
        # 1) if the drone fell to the ground
        print("self.alt ", str(self.alt))
        # if the drone falls, the episode is truncated
        if self.alt < 0.1:  # alt is in meters
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "fall"
        # 2) if the drone reach a corner cone
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.truncated = False
            self.corner = self.CORNERS[np.argmin(distance)]
        # 3) it accumulates too much score without reaching the corner
        if (self.episode_score <= self.MIN_EPISODE_SCORE or
            self.episode_score >= self.MAX_EPISODE_SCORE):
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "score"
        if self.episode_step > self.max_episode_steps:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "max_steps"
        # Webots bug, webots freezes
        # if done:
        #     # Take a picture of the scene
        #     # Nombre del archivo y calidad (0-100)
        #     self.exportImage(str(self.screenshoot_counter)+"-captura_pantalla.png", 100)
        #     self.screenshoot_counter += 1
        return done

    def get_info_keywords(self):
        return tuple(["x_init", "y_init", "closest_corner", "is_success",
                      "corner", "height", "dist_min_target"])

    def get_info(self):
        """
        Method call by step function
        """
        distance = self.get_distance_to_corners()
        self.closest_corner = self.CORNERS[np.argmin(distance)]
        return {"x_init": self.x_init,
                "y_init": self.y_init,
                "closest_corner": self.closest_corner,
                "is_success": self.is_success,
                "corner": self.corner,
                "height": self.alt,
                "dist_min_target": self.dist_min_target}

    def get_reward(self, action=6):
        """
        Modular reward function for SimpleCornerEnv.
        Combines distance to target, obstacle proximity, smooth velocities,
        and penalization for long episodes or falls.
        """
        # Parámetros de pesos (ajustables)
        w_1 = 3000  # Peso para acercamiento al objetivo
        w_bonus = 30  # Bonus por alcanzar el objetivo
        w_2 = 1000  # Peso para proximidad a obstáculos
        w_3 = 20  # Peso para suavidad en velocidades
        w_4 = 0.009  # Penalización por duración del episodio
        w_5 = 50  # Penalización por caída
        k = 5  # Escala exponencial para la distancia al objetivo
        gamma = 10  # Escala exponencial para obstáculos

        reward = 0

        # 1) Recompensa por distancia al objetivo
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("self.dist_min_target " + str(self.dist_min_target))
        r_distance = w_1 * (np.exp(-k * self.prev_dist_min_target) - np.exp(-k * self.dist_min_target))
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            r_distance += w_bonus
        reward += r_distance
        print("r_distance " + str(r_distance))
        self.prev_dist_min_target = self.dist_min_target
        # 2) Penalización por proximidad a obstáculos
        obs_min = np.array(
            [self.dist_front, self.dist_back, self.dist_right, self.dist_left]
        ).min()
        print("obs_min " + str(obs_min))
        r_obstacles = -w_2 * np.exp(-gamma * (obs_min / 1000))
        reward += r_obstacles
        print("r_obstacles " + str(r_obstacles))
        # 3) Penalización por cambios bruscos de velocidades
        if self.past_lineal_velocity is None:
            self.past_lineal_velocity = self.lineal_velocity
        if self.past_angular_velocity is None:
            self.past_angular_velocity = self.angular_velocity
        r_velocity = -w_3 * (
                np.linalg.norm(self.lineal_velocity - self.past_lineal_velocity) +
                np.linalg.norm(self.angular_velocity - self.past_angular_velocity)
        )
        self.past_lineal_velocity = self.lineal_velocity
        self.past_angular_velocity = self.angular_velocity
        print("self.lineal_velocity " + str(self.lineal_velocity))
        print("self.angular_velocity " + str(self.angular_velocity))
        reward += r_velocity
        print("r_velocity " + str(r_velocity))

        # 4) Penalización por duración del episodio
        r_time = -w_4 * self.episode_step
        reward += r_time
        print("r_time " + str(r_time))
        # 5) Penalización por caída
        if self.alt < 0.1:
            reward = -w_5
        print("reward " + str(reward))

        # Actualización de valores previos
        self.prev_dist_min_target = self.dist_min_target

        # Acumulamos la recompensa total
        self.episode_score += reward

        # Normalización (opcional si es necesario)
        normalized_reward = normalize_to_range(reward, -60, 50, -1, 1)

        return normalized_reward

    def get_reward_0001(self, action=6):
        """
                Reward function for CornerEnv in Webots.
                Penalizes greater distance to the objective and smaller distance to walls.
                """
        reward = 0
        # if self.past_angular_velocity is None:
        #    self.past_angular_velocity = self.angular_velocity
        # if self.past_lineal_velocity is None:
        #    self.past_lineal_velocity = self.lineal_velocity
        # np.linalg.norm(state)
        status = - np.sqrt((self.lineal_velocity ** 2).sum())
        print("STATUS vel lineal " + str(status))
        status -= np.sqrt((self.angular_velocity ** 2).sum())
        print("DEBUG  status vel ANGULAR " + str(status))
        if self.status is None:
            self.status = status
        else:
            reward += status - self.status

        print("reward status " + str(reward))
        self.status = status
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("Dist MIN TO TARGET: " + str(self.dist_min_target))

        # 1) drone reaches the target
        d = 3000 * (self.prev_dist_min_target - self.dist_min_target)
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 30
        # 5) otherwise, penalize if it goes away from the target
        else:
            print("Dist to target reward: " + str(d))
            reward += d
            # 2) penalize if the drone is too close any wall
            # Calculate the minimum distance to the walls
            obs_min = np.array(
                [self.dist_front, self.dist_back,
                 self.dist_right, self.dist_left]).min()
            # print in meters
            print("obstacle distance/1000 [m]: " + str(obs_min / 1000))
            # 2) penalize the drone if it is too close to an obstacle
            if obs_min <= self.MIN_DIST_OBSTACLES:
                reward -= 10
            # 4) if the drone falls, the episode is truncated
            if self.alt < 0.1:
                reward = -30

        self.prev_dist_min_target = self.dist_min_target
        print("Final reward " + str(reward))
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -30, 30, -1, 1)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward



class CornerEnvShaped20(CornerEnv):
    def __init__(self, max_episode_steps=10_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)

    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("DEBUG  dist_min " + str(self.dist_min_target))
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        # print in meters
        print(obs_min / 100)
        # 1) drone reaches the target
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 20
        # 2) penalize the drone if it is too close to an obstacle
        if obs_min <= self.MIN_DIST_OBSTACLES:
            reward = -10
        # 4) if the drone falls, the episode is truncated
        if self.alt < 0.1:
            reward = -20
        # 5) otherwise
        if reward == 0:
            reward = 1800 * (self.prev_dist_min_target - self.dist_min_target)
        self.prev_dist_min_target = self.dist_min_target
        print("Reward value " + str(reward))
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -20, 20, -1, 1)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

class CornerEnvShapedReward(CornerEnv):
    def __init__(self, max_episode_steps=10_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)
        self.status = None
        # a vector (vx,vy,vz)
        self.past_lineal_velocity = None
        # a vector (omega_x,omega_y,omega_z)
        self.past_angular_velocity = None
    def _initialization(self):
        super()._initialization()
        self.past_lineal_velocity = None
        self.past_angular_velocity = None
        self.status = None

    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        #if self.past_angular_velocity is None:
        #    self.past_angular_velocity = self.angular_velocity
        #if self.past_lineal_velocity is None:
        #    self.past_lineal_velocity = self.lineal_velocity
        # np.linalg.norm(state)
        status = - np.sqrt((self.lineal_velocity ** 2).sum())
        print("STATUS vel lineal " + str(status))
        status -= np.sqrt((self.angular_velocity ** 2).sum())
        print("DEBUG  status vel ANGULAR " + str(status))
        if self.status is None:
            self.status = status
        else:
            reward += status - self.status

        print("reward status " + str(reward))
        self.status = status
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("Dist MIN TO TARGET: " + str(self.dist_min_target))

        # 1) drone reaches the target
        d = 3000 * (self.prev_dist_min_target - self.dist_min_target)
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 20
        # 5) otherwise, penalize if it goes away from the target
        else:
            print("Dist to target reward: " + str(d))
            reward += d
            # 2) penalize if the drone is too close any wall
            # Calculate the minimum distance to the walls
            obs_min = np.array(
                [self.dist_front, self.dist_back,
                 self.dist_right, self.dist_left]).min()
            # print in meters
            print("obstacle distance/1000 [m]: " + str(obs_min / 1000))
            # 2) penalize the drone if it is too close to an obstacle
            if obs_min <= self.MIN_DIST_OBSTACLES:
                reward -= 10
            # 4) if the drone falls, the episode is truncated
            if self.alt < 0.1:
                reward = -20

        self.prev_dist_min_target = self.dist_min_target
        print("Final reward " + str(reward))
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -20, 20, -1, 1)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

class CornerEnvShapedDirection(CornerEnv):
    def __init__(self, max_episode_steps=30_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)

    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("DEBUG  dist_min " + str(self.dist_min_target))
        # 1) penalize if the drone is not near a target by its distance
        pen_dist= self.penalization_distance_to_target(self.dist_min_target)
        print("DEBUG  pen_dist " + str(pen_dist))
        # get the closest target
        targ = self.TARGET_LIST[np.argmin(distance)]
        pen_dir= self.get_direction_alignment(self.robot,targ)
        print("DEBUG  pen_dist " + str(pen_dir))
        r_dist = 1000*(self.prev_dist_min_target - pen_dist)
        print("DEBUG  r_dist " + str(r_dist))
        r_dir = 1000*(self.prev_dir_target-pen_dir)
        print("DEBUG  r_DIR " + str(r_dir))
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        # print in meters
        print(obs_min / 100)
        # 1) drone reaches the target
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 10
        # 2) penalize the drone if it is too close to an obstacle
        if obs_min <= self.MIN_DIST_OBSTACLES:
            reward = -5
        # 4) if the drone falls, the episode is truncated
        if self.alt < 0.1:
            reward = -10
        # 5) otherwise
        if reward == 0:
            reward = r_dir + r_dist
        print("Reward value " + str(reward))
        self.prev_dist_min_target = self.dist_min_target
        self.prev_dir_target = pen_dir
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -20, 20, -1, 1)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

class SimpleCornerEnvShaped20(DroneRobotSupervisor):
    """
    A Crazyflie base-pilot with the mission of reaching a target
    in a square room.
    """
    CORNERS = ["cone_1"]
    TARGET_LIST = []
    # 15 cm minimal distance to target
    DISTANCE_THRESHOLD = 0.15
    # 100 mm
    MIN_DIST_OBSTACLES = 100
    FILE_NAME_ROTATION = "rotation.txt"
    FILE_NAME_LOCATION = "location.txt"
    MIN_EPISODE_SCORE = -5_000
    MAX_EPISODE_SCORE = 5_000

    def __init__(self, max_episode_steps=20_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.corner = None
        self.x_init = None
        self.y_init = None
        self.status = None
        self.closest_corner = None
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")

        self._init_targets()
        self._initialization()

    def _init_targets(self):
        # 1) get TARGET info
        self.corner = None
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        for name in self.CORNERS:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)

    def put_drone_in_the_center(self):
        position_field = self.robot.getField("translation")
        z = position_field.getSFVec3f()[2]  # Mantener la coordenada Z actual
        nueva_posicion = [0, 0, z]
        self.x_init = nueva_posicion[0]
        self.y_init = nueva_posicion[1]
        position_field.setSFVec3f(nueva_posicion)

    def _initialization(self):
        super()._initialization()
        print("DEBUG episode score "+str(self.episode_score))
        self.dist_min_target = 0
        self.corner = None
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.status = None
        self.put_drone_in_the_center()
        #self._init_rotation()

    def get_distance_to_corners(self):
        d = []
        for t in self.TARGET_LIST:
            d.append(get_distance_from_target(self.robot, t))
        #print(d)
        d = np.array(d)
        return d

    def is_done(self):
        """
        Method call by step function to know if the episode ends.
        Return True when:
         * the drone reach a target corner
         * the drone falls
         * reward reaches a threshold
         """
        done = False
        # 1) if the drone fell to the ground
        print("self.alt ", str(self.alt))
        # if the drone falls, the episode is truncated
        if self.alt < 0.1:  # alt is in meters
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "fall"
        # 2) if the drone reach a corner cone
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.truncated = False
            self.corner = self.CORNERS[np.argmin(distance)]
        # 3) it accumulates too much score without reaching the corner
        if (self.episode_score <= self.MIN_EPISODE_SCORE or
            self.episode_score >= self.MAX_EPISODE_SCORE):
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "score"
        if self.episode_step > self.max_episode_steps:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "max_steps"
        # Webots bug, webots freezes
        # if done:
        #     # Take a picture of the scene
        #     # Nombre del archivo y calidad (0-100)
        #     self.exportImage(str(self.screenshoot_counter)+"-captura_pantalla.png", 100)
        #     self.screenshoot_counter += 1
        return done

    def get_info_keywords(self):
        return tuple(["x_init", "y_init", "closest_corner", "is_success",
                      "corner", "height", "dist_min_target"])

    def get_info(self):
        """
        Method call by step function
        """
        distance = self.get_distance_to_corners()
        self.closest_corner = self.CORNERS[np.argmin(distance)]
        return {"x_init": self.x_init,
                "y_init": self.y_init,
                "closest_corner": self.closest_corner,
                "is_success": self.is_success,
                "corner": self.corner,
                "height": self.alt,
                "dist_min_target": self.dist_min_target}

    def walls_proximity_penalization(self, x):
        #return -10 * np.exp(-(x/0.2))
        # return -10 * np.exp(-(x / 0.4))
        return -7 * np.exp(-(x / 0.1))

    def penalization_distance_to_target(self, x, alpha=5, beta=1):
        #print(f"Tipo de x: {type(x)}, Valor de x: {x}")
        return np.exp(-alpha * x) - (beta * x)


    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("dist_min " + str(self.dist_min_target))
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        # print in meters
        print("obstacle dist : " + str(obs_min / 1000))
        # 1) drone reaches the target
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 20
        # 2) penalize the drone if it is too close to an obstacle
        if obs_min <= self.MIN_DIST_OBSTACLES:
            reward = -10
        # 4) if the drone falls, the episode is truncated
        if self.alt < 0.1:
            reward = -20
        # 5) otherwise
        if reward == 0:
            reward = 2000 * (self.prev_dist_min_target - self.dist_min_target)
        self.prev_dist_min_target = self.dist_min_target
        print("Reward value " + str(reward))
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -20, 20, -1, 1)
        print("Normalized reward " + str(normalized_reward))
        return normalized_reward
    def get_reward_old(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # if self.past_angular_velocity is None:
        #    self.past_angular_velocity = self.angular_velocity
        # if self.past_lineal_velocity is None:
        #    self.past_lineal_velocity = self.lineal_velocity
        # np.linalg.norm(state)
        status = - np.sqrt((self.lineal_velocity ** 2).sum())
        status -= np.sqrt((self.angular_velocity ** 2).sum())
        print("DEBUG  status " + str(status))
        if self.status is None:
            self.status = status
        else:
            reward += status - self.status
        print("REWARD status " + str(reward))
        self.status = status
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("Dist to TARGET: " + str(self.dist_min_target))
        # 1) drone reaches the target
        d = 10_000 * (self.prev_dist_min_target - self.dist_min_target)
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 30
        # 5) otherwise, penalize if it goes away from the target
        else:
            print("REWARD d: " + str(d))
            reward += d
            # 2) penalize if the drone is too close any wall
            # Calculate the minimum distance to the walls
            obs_min = np.array(
                [self.dist_front, self.dist_back,
                 self.dist_right, self.dist_left]).min()
            # print in meters
            print("obstacle distance/1000 [m]: " + str(obs_min / 1000))
            # 2) penalize the drone if it is too close to an obstacle
            if obs_min <= self.MIN_DIST_OBSTACLES:
                reward -= 20
            # 4) if the drone falls, the episode is truncated
            if self.alt < 0.1:
                reward = -30
        self.prev_dist_min_target = self.dist_min_target
        print("Final reward " + str(reward))
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -30, 30, -1, 1)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward
class SimpleCornerEnvRF(DroneRobotSupervisor):
    """
    A Crazyflie base-pilot with the mission of reaching the cone_1
    target.
    Drone start always in the same position, the center of the room.
    """
    CORNERS = ["cone_1"]
    TARGET_LIST = []
    # 15 cm minimal distance to target
    DISTANCE_THRESHOLD = 0.15
    # 100 mm = 10 cm
    MIN_DIST_OBSTACLES = 100
    MIN_EPISODE_SCORE = -1_000
    MAX_EPISODE_SCORE = 1_000

    def __init__(self, max_episode_steps=10_000):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps)
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.corner = None
        self.x_init = None
        self.y_init = None
        self.status = None
        self.closest_corner = None
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")
        self._init_targets()
        self._initialization()

    def _init_targets(self):
        # 1) get TARGET info
        self.corner = None
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        for name in self.CORNERS:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)

    def put_drone_in_the_center(self):
        position_field = self.robot.getField("translation")
        z = position_field.getSFVec3f()[2]  # Mantener la coordenada Z actual
        nueva_posicion = [0, 0, z]
        self.x_init = nueva_posicion[0]
        self.y_init = nueva_posicion[1]
        position_field.setSFVec3f(nueva_posicion)

    def _initialization(self):
        super()._initialization()
        print("DEBUG episode score "+str(self.episode_score))
        self.dist_min_target = 0
        self.corner = None
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.status = None
        self.put_drone_in_the_center()

    def get_distance_to_corners(self):
        d = []
        for t in self.TARGET_LIST:
            d.append(get_distance_from_target(self.robot, t))
        #print(d)
        d = np.array(d)
        return d

    def is_done(self):
        """
        Method call by step function to know if the episode ends.
        Return True when:
         * the drone reach a target corner
         * the drone falls
         * reward reaches a threshold
         """
        done = False
        # 1) if the drone fell to the ground
        print("self.alt ", str(self.alt))
        # if the drone falls, the episode is truncated
        if self.alt < 0.1:  # alt is in meters
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "fall"
        # 2) if the drone reach a corner cone
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.truncated = False
            self.corner = self.CORNERS[np.argmin(distance)]
        # 3) it accumulates too much score without reaching the corner
        if (self.episode_score <= self.MIN_EPISODE_SCORE or
            self.episode_score >= self.MAX_EPISODE_SCORE):
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "score"
        if self.episode_step > self.max_episode_steps:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "max_steps"
        # Webots bug, webots freezes
        # if done:
        #     # Take a picture of the scene
        #     # Nombre del archivo y calidad (0-100)
        #     self.exportImage(str(self.screenshoot_counter)+"-captura_pantalla.png", 100)
        #     self.screenshoot_counter += 1
        return done

    def get_info_keywords(self):
        return tuple(["x_init", "y_init", "closest_corner", "is_success",
                      "corner", "height", "dist_min_target"])

    def get_info(self):
        """
        Method call by step function
        """
        distance = self.get_distance_to_corners()
        self.closest_corner = self.CORNERS[np.argmin(distance)]
        return {"x_init": self.x_init,
                "y_init": self.y_init,
                "closest_corner": self.closest_corner,
                "is_success": self.is_success,
                "corner": self.corner,
                "height": self.alt,
                "dist_min_target": self.dist_min_target}

    def walls_proximity_penalization(self, x):
        #return -10 * np.exp(-(x/0.2))
        return -10 * np.exp(-(x / 0.04))
        #return -7 * np.exp(-(x / 0.1))

    #def penalization_distance_to_target(self, x, alpha=5, beta=1):
        #print(f"Tipo de x: {type(x)}, Valor de x: {x}")
    #    return np.exp(-alpha * x) - (beta * x)

    def penalization_distance_to_target(self, x):
        return 20 * np.exp(-5 * x)

    def episode_steps_penalization(self, x):
        return -1 * np.log(1 + x)

    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("DISTANCIA a TARGET " + str(self.dist_min_target))
        # 1) reward dist to target
        reward+= self.penalization_distance_to_target(self.dist_min_target)
        print("Distance reward: "+str(reward))
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        # print in meters
        aux = obs_min / 1000
        print("OBSTACULO dist : " + str(aux))
        print("Obstacle penalization: " + str(self.walls_proximity_penalization(aux)))
        reward += self.walls_proximity_penalization(aux)
        self.prev_dist_min_target = self.dist_min_target
        #aux = self.episode_steps_penalization(self.episode_step)
        #reward += aux
        #print("Episode len penalization: " + str(aux))
        print("Reward: " + str(reward))
        # Normalize reward
        normalized_reward = np.tanh(reward)
        #normalized_reward = normalize_to_range(reward, -10, 10, -1, 1)
        print("Normalized reward " + str(normalized_reward))
        # update cumulative reward
        self.episode_score += normalized_reward
        print("Cumulative score " + str(self.episode_score))
        return normalized_reward


class SimpleCornerEnvRS10(DroneRobotSupervisor):
    """
    A Crazyflie base-pilot with the mission of reaching a target
    in a square room.
    """
    CORNERS = ["cone_1"]
    TARGET_LIST = []
    # 15 cm minimal distance to target
    DISTANCE_THRESHOLD = 0.15
    # 100 mm
    MIN_DIST_OBSTACLES = 100
    FILE_NAME_ROTATION = "rotation.txt"
    FILE_NAME_LOCATION = "location.txt"
    MIN_EPISODE_SCORE = -1_000
    MAX_EPISODE_SCORE = 1_000

    def __init__(self, max_episode_steps=20_000, pilot = None):
        # call DroneRobotSupervisor constructor
        super().__init__(max_episode_steps, pilot)
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.corner = None
        self.x_init = None
        self.y_init = None
        self.status = None
        self.closest_corner = None
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")

        self._init_targets()
        self._initialization()

    def _init_targets(self):
        # 1) get TARGET info
        self.corner = None
        self.dist_min_target = 0
        self.prev_dist_min_target = 0
        for name in self.CORNERS:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)

    def put_drone_in_the_center(self):
        position_field = self.robot.getField("translation")
        z = position_field.getSFVec3f()[2]  # Mantener la coordenada Z actual
        nueva_posicion = [0, 0, z]
        self.x_init = nueva_posicion[0]
        self.y_init = nueva_posicion[1]
        position_field.setSFVec3f(nueva_posicion)

    def _initialization(self):
        super()._initialization()
        print("DEBUG episode score "+str(self.episode_score))
        self.dist_min_target = 0
        self.corner = None
        self.prev_dist_min_target = 0
        self.prev_dir_target = 0
        self.status = None
        self.put_drone_in_the_center()
        #self._init_rotation()

    def get_distance_to_corners(self):
        d = []
        for t in self.TARGET_LIST:
            d.append(get_distance_from_target(self.robot, t))
        #print(d)
        d = np.array(d)
        return d

    def is_done(self):
        """
        Method call by step function to know if the episode ends.
        Return True when:
         * the drone reach a target corner
         * the drone falls
         * reward reaches a threshold
         """
        done = False
        # 1) if the drone fell to the ground
        print("self.alt ", str(self.alt))
        # if the drone falls, the episode is truncated
        if self.alt < 0.1:  # alt is in meters
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "fall"
        # 2) if the drone reach a corner cone
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.truncated = False
            self.corner = self.CORNERS[np.argmin(distance)]
        # 3) it accumulates too much score without reaching the corner
        if (self.episode_score <= self.MIN_EPISODE_SCORE or
            self.episode_score >= self.MAX_EPISODE_SCORE):
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "score"
        if self.episode_step > self.max_episode_steps:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "max_steps"
        # Webots bug, webots freezes
        # if done:
        #     # Take a picture of the scene
        #     # Nombre del archivo y calidad (0-100)
        #     self.exportImage(str(self.screenshoot_counter)+"-captura_pantalla.png", 100)
        #     self.screenshoot_counter += 1
        return done

    def get_info_keywords(self):
        return tuple(["x_init", "y_init", "closest_corner", "is_success",
                      "corner", "height", "dist_min_target"])

    def get_info(self):
        """
        Method call by step function
        """
        distance = self.get_distance_to_corners()
        self.closest_corner = self.CORNERS[np.argmin(distance)]
        return {"x_init": self.x_init,
                "y_init": self.y_init,
                "closest_corner": self.closest_corner,
                "is_success": self.is_success,
                "corner": self.corner,
                "height": self.alt,
                "dist_min_target": self.dist_min_target}

    def walls_proximity_penalization(self, x):
        #return -10 * np.exp(-(x/0.2))
        # return -10 * np.exp(-(x / 0.4))
        return 50 * x - 10

    def penalization_distance_to_target(self, x):
        #print(f"Tipo de x: {type(x)}, Valor de x: {x}")
        return 1 / x


    def get_reward(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("Dist to corner1 " + str(self.dist_min_target))
        # 2) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        # print in meters
        print("Obstacle dist : " + str(obs_min / 1000))
        # 1) drone reaches the target
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 10
        # 2) if the drone falls, the episode is truncated
        if self.alt < 0.1:
            reward = -10
        # 3) episode truncated
        if (self.episode_score <= self.MIN_EPISODE_SCORE or
            self.episode_score >= self.MAX_EPISODE_SCORE) or (self.episode_step > self.max_episode_steps):
            reward = -10
        # 5) otherwise
        if reward == 0:
            # 5.1) penalize the drone if it is too close to an obstacle
            if obs_min <= self.MIN_DIST_OBSTACLES:
                reward += self.walls_proximity_penalization(obs_min/1000)
            # 5.2) reward function by distance to the target
            reward += self.penalization_distance_to_target(self.dist_min_target)
        self.prev_dist_min_target = self.dist_min_target
        print("Reward value " + str(reward))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -10, 10, -1, 1)
        print("Normalized reward " + str(normalized_reward))
        # update cumulative reward
        self.episode_score += normalized_reward
        print("Episode cumulative score " + str(self.episode_score))
        return normalized_reward
    def get_reward_old(self, action=6):
        """
        Reward function for CornerEnv in Webots.
        Penalizes greater distance to the objective and smaller distance to walls.
        """
        reward = 0
        # if self.past_angular_velocity is None:
        #    self.past_angular_velocity = self.angular_velocity
        # if self.past_lineal_velocity is None:
        #    self.past_lineal_velocity = self.lineal_velocity
        # np.linalg.norm(state)
        status = - np.sqrt((self.lineal_velocity ** 2).sum())
        status -= np.sqrt((self.angular_velocity ** 2).sum())
        print("DEBUG  status " + str(status))
        if self.status is None:
            self.status = status
        else:
            reward += status - self.status
        print("REWARD status " + str(reward))
        self.status = status
        # Get min distance to corners
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("Dist to TARGET: " + str(self.dist_min_target))
        # 1) drone reaches the target
        d = 10_000 * (self.prev_dist_min_target - self.dist_min_target)
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 30
        # 5) otherwise, penalize if it goes away from the target
        else:
            print("REWARD d: " + str(d))
            reward += d
            # 2) penalize if the drone is too close any wall
            # Calculate the minimum distance to the walls
            obs_min = np.array(
                [self.dist_front, self.dist_back,
                 self.dist_right, self.dist_left]).min()
            # print in meters
            print("obstacle distance/1000 [m]: " + str(obs_min / 1000))
            # 2) penalize the drone if it is too close to an obstacle
            if obs_min <= self.MIN_DIST_OBSTACLES:
                reward -= 30
            # 4) if the drone falls, the episode is truncated
            if self.alt < 0.1:
                reward = -30
        self.prev_dist_min_target = self.dist_min_target
        print("Final reward " + str(reward))
        # update cumulative reward
        self.episode_score += reward
        print("Episode cumulative score " + str(self.episode_score))
        # Normalize reward
        normalized_reward = normalize_to_range(reward, -30, 30, -1, 1)
        print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward
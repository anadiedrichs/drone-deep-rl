from copilot.CrazyflieDrone import *
from copilot.rotation import *
import numpy as np
class CornerEnv(DroneRobotSupervisor):
    """
    A Crazyflie base-pilot with the mission of reaching a target
    in a square room.
    """
    CORNERS = ["cone_1", "cone_2", "cone_3","cone_4"]
    TARGET_LIST = []
    # 20 cm
    DISTANCE_THRESHOLD = 0.20
    # 100 mm
    MIN_DIST_OBSTACLES = 100

    FILE_NAME = "rotation.txt"
    def __init__(self):
        # call DroneRobotSupervisor constructor
        super().__init__()
        self.robot = self.getFromDef('crazyflie')
        self.corner = None
        self.dist_min_target = 0
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")
        for name in self.CORNERS:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)
        # ROTATION CHANGE
        # Leer la rotación desde el archivo (que ahora está actualizada)
        rotacion = leer_rotacion_desde_archivo(self.FILE_NAME)
        if rotacion:
            print(f"Rotación leída: {rotacion}")

        # Asignar la rotación al drone
        self.robot.getField("rotation").setSFRotation(rotacion)

        # Incrementar la rotación en el eje Z y actualizar el archivo
        incrementar_rotacion_en_z(self.FILE_NAME)

    def get_distance_to_corners(self):
        d =[]
        for t in self.TARGET_LIST:
            d.append(get_distance_from_target(self.robot,t))
        print(d)
        d = np.array(d)
        return d

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

    def get_reward(self, action=6):

        """
         Method call by step function.
         If the distance to the target >= DISTANCE_THRESHOLD,
            penalize by - 0.1 * dist_min( to one of the corners)
         Otherwise, apply a stepped reward function.
         action parameter is not used in the current implementation
        """
        reward = 0
        action = int(action)

        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        print("DEBUG  dist_min " + str(self.dist_min_target))
        # 1) reward according to the distance to one of the corners
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            reward = 50
        elif self.dist_min_target <= self.DISTANCE_THRESHOLD*2:
            reward = 40
        elif self.dist_min_target <= self.DISTANCE_THRESHOLD*3:
            reward = 30
        elif self.dist_min_target <= self.DISTANCE_THRESHOLD*4:
            reward = 20
        else:
            # 2) penalize if the drone is not near a corner
            reward = -1 * self.dist_min_target
        # 3) penalize if the drone is too close any wall
        # Calculate the minimum distance to the walls
        obs_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        if obs_min <= self.MIN_DIST_OBSTACLES:
            # Penaliza más cuanto más cerca está de la pared
            reward -= (self.MIN_DIST_OBSTACLES - obs_min)
        # 4) big penalty if it takes to long to reach a corner
        if self.episode_score <= -100_000 and self.dist_min_target > self.DISTANCE_THRESHOLD:
            reward -= -10
        print("Reward value " + str(reward))
        # update cummulative reward
        self.episode_score += reward
        print("Episode score " + str(self.episode_score))
        # Calcular valores mínimo y máximo de recompensa
        # Depende de las dimensiones del escenario
        min_reward = -12.1  #
        max_reward = 50  #
        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, min_reward, max_reward, -1, 1)
        # print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

    def is_done(self):
        """
        Method call by step function
        Return True when:
         * the drone reach a corner
        """
        done = False
        # 1) if the drone fell to the ground
        print("ALTURA ", str(self.alt))
        # alt is in meters
        if self.alt < 0.1:
            done = True
            self.terminated = True
            self.is_success = False
            self.corner = "fall"
        # 2) if the drone reach a corner cone
        distance = self.get_distance_to_corners()
        self.dist_min_target = distance.min()
        if self.dist_min_target <= self.DISTANCE_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.corner = self.CORNERS[np.argmin(distance)]
        # 3) it accumulates too much score without reaching the corner
        if self.episode_score <= -100_000 or self.episode_score > 100_000:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True
            self.corner = "none"
        return done

    def get_info(self):
        """
        Method call by step function
        """
        return {"is_success": self.is_success,
                "corner":self.corner,
                "height":self.alt,
                "dist_min_target":self.dist_min_target}

    def get_info_keywords(self):
        return tuple(["is_success", "corner","height", "dist_min_target"])

# TODO heredar del escenario anterior ??
class TargetAndObstaclesEnv(DroneRobotSupervisor):
    """
    A Crazyflie quadcopter simulation environment.
    Mission:
    * Reach a target in a square room.
    * Avoid obstacle
    """
    TARGET_LIST = ["cone_1", "cone_2", "cone_3"]
    # 20 cm
    FIND_THRESHOLD = 0.2
    # 10 cm
    MIN_DIST_OBSTACLES = 100

    def __init__(self, target_name=None, seed=5):
        super().__init__()

        self.target_name = ""
        if target_name is None:
            # choose a target randomly from TARGET_LIST
            random.seed(seed)
            self.target_name = self.TARGET_LIST[random.randint(0, len(self.TARGET_LIST) - 1)]
        else:
            # setup target
            self.target_name = target_name
        self.target = self.getFromDef(self.target_name)
        self.robot = self.getFromDef('crazyflie')
        if self.target is None:
            raise ValueError("Check target_name or see DEF in world config.")
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")
    def get_info(self):

        """

        """
        return {"is_success": self.is_success,
                "target_name": self.target_name,
                }

    def get_reward(self, action=6):
        """
        If the drone reach the target, gets a big reward.
        Otherwise:
        * penalize using the distance to the target
        * penalize if the drone is close to an obstacle
        """
        reward= 0
        # Calculate the minimum distance to the walls
        dist_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        print("DEBUG  dist_min " + str(dist_min))
        # if the drone reach the target
        distance = get_distance_from_target(self.robot,self.target)
        print("DEBUG  distance " + str(distance)+" to target "+self.target_name)
        #1) if reach the target
        print("DEBUG  dist_min " + str(dist_min))
        # 1) reward according to the distance to one of the corners
        if distance <= self.FIND_THRESHOLD:
            reward = 50
        elif distance <= self.FIND_THRESHOLD * 2:
            reward = 40
        elif distance <= self.FIND_THRESHOLD * 3:
            reward = 30
        elif distance <= self.FIND_THRESHOLD * 4:
            reward = 20
        else:
            # 2) penalize if the drone is not near the target
            reward = -1 * distance
            # 3) penalize if the drone is close to an obstacle
            if dist_min <= self.MIN_DIST_OBSTACLES:
                # Penaliza más cuanto más cerca está del obstáculo
                reward -= (self.MIN_DIST_OBSTACLES-dist_min)*1
            # 4) big penalty if it takes to long to reach a corner
            # and it is far away from the target
            if self.episode_score <= -100_000 and distance >= self.FIND_THRESHOLD*2:
                reward -= 10

        print("Reward value " + str(reward))
        # update cummulative reward
        self.episode_score += reward
        print("Episode score " + str(self.episode_score))
        # Calcular valores mínimo y máximo de recompensa
        # Depende de las dimensiones del escenario
        min_reward = -12.1  #
        max_reward = 50  #
        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, min_reward, max_reward, -1, 1)
        return normalized_reward

    def is_done(self):
        """
        Return True when:
         * the drone reach a target
         * The drone fell to the ground
        """
        done = False
        # 1) if the drone fell to the ground
        print("ALTURA ", str(self.alt))
        # alt is in meters
        if self.alt < 0.1:
            done = True
            self.terminated = True
            self.is_success = False
        # 2) if the drone reach the target
        distance = get_distance_from_target(self.robot,self.target)
        if distance <= self.FIND_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
        # it accumulates too much score without reaching the corner
        if self.episode_score <= -100_000:# or self.episode_score > 2_000_000:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True

        return done

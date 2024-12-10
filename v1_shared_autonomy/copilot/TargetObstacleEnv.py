import numpy as np
from copilot.CrazyflieDrone import *
from copilot.rotation import *
# TODO heredar del escenario anterior ??
class TargetAndObstaclesEnv(DroneRobotSupervisor):
    """
    A Crazyflie quadcopter simulation environment.
    Mission:
    * Reach a target in a square room.
    * Avoid obstacle
    """
    CONE = ["cone_1", "cone_2", "cone_3"]
    TARGET_LIST = []
    FILE_LAST_TARGET="last_target.txt"
    # 20 cm
    FIND_THRESHOLD = 0.2
    # 10 cm
    MIN_DIST_OBSTACLES = 100
    FILE_NAME_ROTATION = "rotation.txt"
    def __init__(self, target_name=None, seed=5):
        super().__init__()
        self.robot = self.getFromDef('crazyflie')
        if self.robot is None:
            raise ValueError("Check drone name should be crazyflie, see DEF in world config.")
        self.target_name = ""
        self.dist_target = 0
        self._initialization()

    def _init_targets(self):
        # 1) get TARGET info
        self.TARGET_LIST = []
        for name in self.CONE:
            t = self.getFromDef(name)
            if t is None:
                raise ValueError("Check target_name or see DEF in world config.")
            self.TARGET_LIST.append(t)
        if len(self.TARGET_LIST) != len(self.CONE):
            raise ValueError("The len of CONE and TARGET_LIST should be the same")

    def _initialization(self):
        super()._initialization()
        self.dist_target = 0
        self.target_name = ""
        self._init_rotation()
        self._init_targets()
        self._choose_target()

    def _choose_target(self):
        try:
            with open(self.FILE_LAST_TARGET, 'r') as file:
                target_index = int(file.read().strip())
        except FileNotFoundError:
            target_index = 0
        # Ensure target_index is within the bounds of TARGET_LIST
        if target_index >= len(self.CONE):
            target_index = 0
        print("target_index "+str(target_index))
        print(f"target_index: {target_index}, len(TARGET_LIST): {len(self.TARGET_LIST)}")

        # Set target_name and target
        self.target_name = self.CONE[target_index]
        self.target = self.TARGET_LIST[target_index]
        # Increment target_index and write back to file
        target_index = (target_index + 1) % len(self.CONE)
        with open(self.FILE_LAST_TARGET, 'w') as file:
            file.write(str(target_index))
    def _init_rotation(self):
        # 2) ROTATION CHANGE
        # Leer la rotación desde el archivo (que ahora está actualizada)
        rotacion = leer_rotacion_desde_archivo(self.FILE_NAME_ROTATION)
        if rotacion:
            print(f"Rotación leída: {rotacion}")
        # Asignar la rotación al drone
        self.robot.getField("rotation").setSFRotation(rotacion)
        # Incrementar la rotación en el eje Z y actualizar el archivo
        incrementar_rotacion_en_z(self.FILE_NAME_ROTATION)

    def get_info_keywords(self):
        return tuple(["is_success", "target_name", "height", "dist_target"])

    def get_info(self):
        """
        Method call by step function
        """

        return {"is_success": self.is_success,
                "target_name": self.target_name,
                "height": self.alt,
                "dist_target": self.dist_target}

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
        self.dist_target = get_distance_from_target(self.robot,self.target)
        print("DEBUG  distance " + str(self.dist_target)+" to target "+self.target_name)

        # 1) reward according to the distance to the target
        if self.dist_target <= self.FIND_THRESHOLD:
            reward = 50
        elif self.dist_target <= self.FIND_THRESHOLD * 2:
            reward = 40
        elif self.dist_target <= self.FIND_THRESHOLD * 3:
            reward = 30
        elif self.dist_target <= self.FIND_THRESHOLD * 4:
            reward = 20
        else:
            # 2) penalize if the drone is not near the target
            reward = -1 * self.dist_target
            # 3) penalize if the drone is close to an obstacle
            if dist_min <= self.MIN_DIST_OBSTACLES:
                # Penaliza más cuanto más cerca está del obstáculo
                reward -= (self.MIN_DIST_OBSTACLES-dist_min)*1
            # 4) big penalty if it takes to long to reach a corner
            # and it is far away from the target
            if self.episode_score <= -100_000 and self.dist_target >= self.FIND_THRESHOLD*2:
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
            #self.target_name="fall"

        # 2) if the drone reach the target
        self.dist_target = get_distance_from_target(self.robot, self.target)
        if self.dist_target <= self.FIND_THRESHOLD:
            done = True
            self.terminated = True
            self.is_success = True
            self.truncated = False
        # it accumulates too much score without reaching the corner
        if self.episode_score <= -100_000: # or self.episode_score > 2_000_000:
            done = True
            self.terminated = True
            self.is_success = False
            self.truncated = True

        return done

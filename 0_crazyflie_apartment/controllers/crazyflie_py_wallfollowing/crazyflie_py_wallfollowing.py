# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2023 Bitcraze


"""
file: crazyflie_py_wallfollowing.py

Controls the crazyflie and implements a wall following method in webots in Python

Author:   Kimberly McGuire (Bitcraze AB)
"""


from controller import Robot
from controller import Keyboard

from math import cos, sin

from pid_controller import pid_velocity_fixed_height_controller
from wall_following import WallFollowing

FLYING_ATTITUDE = 1

if __name__ == '__main__':

    robot = Robot()
    
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    range_front = robot.getDevice("range_front")
    range_front.enable(timestep)
    range_left = robot.getDevice("range_left")
    range_left.enable(timestep)
    range_back = robot.getDevice("range_back")
    range_back.enable(timestep)
    range_right = robot.getDevice("range_right")
    range_right.enable(timestep)

    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Initialize variables

    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE

    wall_following = WallFollowing(angle_value_buffer=0.01, reference_distance_from_wall=0.5,
                                   max_forward_speed=0.3, init_state=WallFollowing.StateWallFollowing.FORWARD)

    autonomous_mode = False

    print("\n")

    print("====== Controls =======\n\n")

    print(" The Crazyflie can be controlled from your keyboard!")
    print(" All controllable movement is in body coordinates")
    print("- Use the up, back, right and left button to move in the horizontal plane")
    print("- Use Q and E to rotate around yaw ")
    print("- Use W and S to go up and down ")

    # Main loop:
    while robot.step(timestep) != -1:

        dt = robot.getTime() - past_time
        actual_state = {}

        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        # Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        key = keyboard.getKey()
        while key > 0:
        
            print("====== SENSORS observations =======\n")
            print(" Roll   " + str(roll) )
            print(" Pitch  " + str(pitch) )
            print(" Yaw    " + str(yaw) )        
            print("Yaw rate: " + str(yaw_rate) )        
            print("x_global: " + str(x_global) )
            print("v_x_global: " + str(v_x_global) )
            print("y_global: " + str(y_global) )
            print("v_y_global: " + str(v_y_global) )        
            print("altitude: " + str(altitude) )
            print("range_front: " + str(range_front_value) )
            print("range_right: " + str(range_right_value) )
            print("range_left: " + str(range_left_value) )
            print("==================================\n")
            
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT:
                sideways_desired -= 0.5
            elif key == Keyboard.LEFT:
                sideways_desired += 0.5
            elif key == ord('Q'):
                yaw_desired = + 1
            elif key == ord('E'):
                yaw_desired = - 1
            elif key == ord('W'):
                height_diff_desired = 0.1
            elif key == ord('S'):
                height_diff_desired = - 0.1
                
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt

        camera_data = camera.getImage()

        # get range in meters
        range_front_value = range_front.getValue() / 1000
        range_right_value = range_right.getValue() / 1000
        range_left_value = range_left.getValue() / 1000
        

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree

class VSSEnvGA(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self, use_fira=False, kp=0.1, kd=0.2, norm_max_speed=1.0):
        super().__init__(field_type=0, n_robots_blue=1, n_robots_yellow=0,
                         time_step=0.025, use_fira=use_fira)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(40, ), dtype=np.float32)
        # print('kd:')
        # print(kd)
        self.point_to_go = 0
        self.sum_error = 0
        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05
        self.kp = kp
        self.kd = kd
        self.last_error = 0
        self.max_speed = 10
        self.steps = 0
        self.ou_actions = []
        self.actual_distance = 0
        self.reset_error = False
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        # print('Environment initialized')
    def calcular_angulo_alvo(self, x_alvo, y_alvo, x_robo, y_robo):
        # Calculate the angle to the target in degrees using arctan2.
        return np.degrees(math.atan2(y_alvo - y_robo, x_alvo - x_robo))

    def diferenca_menor_angulo(self, angulo_alvo, angulo_robo):
        # Calculate the smallest angle difference.
        diff = angulo_alvo - angulo_robo
        diff = (diff + 180) % 360 - 180
        return diff

    def diferenca_menor_angulo2(self, angulo_alvo, angulo_robo):
        # Calculate the smallest angle difference.
        diff = (angulo_robo - angulo_alvo + 180) % 360  - 180
        if diff < -180:
            diff += 360
        return diff
    def navigation(self, target):
        x, y = target
        reversed = False

        robot_angle = self.frame.robots_blue[0].theta
        x_robo = self.frame.robots_blue[0].x
        y_robo = self.frame.robots_blue[0].y
        angle_to_target = self.calcular_angulo_alvo(x, y, x_robo, y_robo)
        # the error will be the smallest angle dist between the robot and the target
        # the robot_angle is in degrees, so we need to convert it to radians
        error = self.diferenca_menor_angulo(angle_to_target, robot_angle)

        # if error > 180:
        #     error -= 360
        # elif error < -180:
        #     error += 360

        if error < 0:
            reversed = True
            # error = -error
        if error > 90:
            reversed = not reversed
            # error = 180 - error

        # the error is now in the range [0, 90]
        motor_speed = (self.kp * error) + (self.kd * (error - self.last_error))
        self.last_error = error
        # if self.reset_error:
        #     error *= 0.6
        self.sum_error += abs(np.deg2rad(error))
        if motor_speed > self.max_speed:
            motor_speed = self.max_speed
        elif motor_speed < -self.max_speed:
            motor_speed = -self.max_speed

        left_wheel_speed = motor_speed
        right_wheel_speed = motor_speed

        if motor_speed > 0:
            left_wheel_speed = motor_speed
            right_wheel_speed = self.max_speed - motor_speed
        else:
            left_wheel_speed = self.max_speed + motor_speed
            right_wheel_speed = -motor_speed
        
        if reversed:
            left_wheel_speed, right_wheel_speed = right_wheel_speed, left_wheel_speed

        return left_wheel_speed, right_wheel_speed

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        for ou in self.ou_actions:
            ou.reset()

        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        if self.steps > 1000:
            done = True

        self.steps += 1
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(
                self.norm_w(self.frame.robots_yellow[i].v_theta)
            )

        return np.array(observation, dtype=np.float32)
    
    def compute_dist(self, x1,y1,x2,y2):
        return math.sqrt((x2-x1)**2+(y2-y1)**2)
    def compute_point(self):
        # in this function we will calculate the point of the target
        # i need to calculate the point of the target based in a square
        # the square starts in the middle of the field and ends in the middle of the goal
        # the target will be the middle of the goal
        # the target changes based in the position of the robot.
        points = [
            (0.5, 0.0),
            (0.5, -0.5),
            (0.0, -0.5),
            (0.0, 0.0)
        ]
        new_point_to_go = self.point_to_go % 4
        if self.compute_dist(self.frame.robots_blue[0].x, self.frame.robots_blue[0].y, points[new_point_to_go][0], points[new_point_to_go][1]) < 0.1:
            self.point_to_go += 1
            new_point_to_go = self.point_to_go % 4
            self.reset_error = True
            # print(points[new_point_to_go])
            # print(self.point_to_go)
            # print(self.compute_dist(self.frame.robots_blue[0].x, self.frame.robots_blue[0].y, points[new_point_to_go][0], points[new_point_to_go][1]))
        return points[new_point_to_go]
    
    def _get_commands(self, actions):
        commands = []
        self.actions = {}
        actions = self.compute_point()
        self.actions[0] = actions
        v_wheel0, v_wheel1 = self.navigation(actions)
        # if self.reset_error:
        #     print('reset error')
        #     print(self.sum_error)
        #     self.reset_error = False
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))

        # Send random commands to the other robots
        for i in range(1, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue+i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):
        reward = -self.sum_error
        
        goal = False
        # if self.frame.robots_blue[0].x > 0.75:
        #     goal = True
        return reward, goal

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=-0.5, y=-0.5)

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=0, y=0, theta=0)

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed , right_wheel_speed

    def __ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0)\
            + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def __move_reward(self):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_ball = ball - robot
        robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty

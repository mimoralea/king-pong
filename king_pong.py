#!/usr/bin/python
from __future__ import division
import numpy as np
import random
import pygame
from shapely.geometry import LineString

# pyGame initialization
FPS = 60
QFPS = 240
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
pygame.init()
FPS_CLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("King Pong")
pygame.font.init()
SCORE_FONT = pygame.font.Font(None, 64)
GAMES_FONT = pygame.font.Font(None, 16)

# Paddle dimensions
PADDLE_WIDTH, PADDLE_HEIGHT = 8, 64
PADDLE_UPPER_SECTION = 3*PADDLE_HEIGHT/8
PADDLE_BOTTOM_SECTION = 5*PADDLE_HEIGHT/8

TOP_SPEED = 5
PADDLE_SPEED = TOP_SPEED
PADDLE_X_DISTANCE, PADDLE_Y_DISTANCE = 16, int(SCREEN_HEIGHT/2)

# Ball
BALL_SIZE = 8


class GameState:
        """
        Game State Representation

        Game state with function to act
        based on user actions.
        """
        def __init__(self, auto_draw = True):
                self.auto_draw = auto_draw
                self.top_speed = TOP_SPEED
                self.reset_positions()
                self.first_to = [1000, 5]
                self.games = [0, 0]
                self.score = [0, 0]
                self.score_changed = False

        def score_last_changed(self):
                """
                Checks if the scores has changed since
                the last time this function was accessed
                """
                current = self.score_changed
                self.score_changed = False
                return current

        def game_over(self):
                """
                The game is over when any player reaches
                the number of games playing to
                """
                return self.games[0] == self.first_to[0] or \
                        self.games[1] == self.first_to[0]

        def reset_positions(self):
                """
                Moves the players to a center position
                and reset the direction and speed of
                the ball randomly within acceptable range.
                """
                self.playerx, self.playery = SCREEN_WIDTH-PADDLE_X_DISTANCE, PADDLE_Y_DISTANCE
                self.cpux, self.cpuy = PADDLE_X_DISTANCE, PADDLE_Y_DISTANCE
                self.ballx, self.bally = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
                self.ball_speed_x = random.choice(
                        range(-self.top_speed+1, -int(2*self.top_speed/3)) +
                        range(int(2*self.top_speed/3), self.top_speed))
                self.ball_speed_y = random.choice(
                        range(-self.top_speed+1, -int(2*self.top_speed/3)) +
                        range(int(2*self.top_speed/3), self.top_speed))

        def frame_step(self, input_actions):
                """
                Moves the state of the game forward
                one step with the given input actions

                input_actions[0] == 1: do nothing
                input_actions[1] == 1: move up
                input_actions[2] == 1: move down

                sum(input_actions) == 1
                """
                pygame.event.pump()

                if sum(input_actions) != 1:
                        raise ValueError('Multiple input actions!')

                # move player
                if input_actions[1] == 1:
                        # player moves up
                        self.playery = np.maximum(0,
                                                  self.playery - self.top_speed)
                elif input_actions[2] == 1:
                        # player moves down
                        self.playery = np.minimum(self.playery + self.top_speed,
                                                  SCREEN_HEIGHT - PADDLE_HEIGHT)

                # move cpu
                if self.cpuy + (PADDLE_HEIGHT/2) > self.bally:
                    self.cpuy = np.maximum(0,
                                           self.cpuy - self.top_speed)
                elif self.cpuy + (PADDLE_HEIGHT/2) < self.bally:
                    self.cpuy = np.minimum(self.cpuy + self.top_speed,
                                           SCREEN_HEIGHT - PADDLE_HEIGHT)

                # move ball get reward the it produced
                reward = self.move_ball()

                # check for losing
                terminal_good = self.ballx <= 0
                terminal_bad = self.ballx + BALL_SIZE >= SCREEN_WIDTH
                terminal = terminal_good or terminal_bad
                if terminal: self.reset_positions()

                self.score[0] += terminal_bad
                self.score[1] += terminal_good

                reward = -1.0 if terminal_bad else 1.0 if terminal_good else reward

                # redraw game onto screen
                SCREEN.fill((0, 0, 0)) # black screen
                pygame.draw.rect(SCREEN, # left 'cpu' player
                                 (255, 255, 255),
                                 (self.cpux, self.cpuy, PADDLE_WIDTH, PADDLE_HEIGHT))
                pygame.draw.rect(SCREEN, # right player
                                 (255, 255, 255),
                                 (self.playerx, self.playery, PADDLE_WIDTH, PADDLE_HEIGHT))
                pygame.draw.rect(SCREEN, # ball
                                 (255, 255, 255),
                                 (self.ballx, self.bally, BALL_SIZE, BALL_SIZE))

                # update pygame
                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                if self.auto_draw: self.complete_drawing()

                if terminal: self.score_changed = True

                # calculate who would be the winner
                if self.score[0] == self.first_to[1]:
                    self.score = [0, 0]
                    self.games[0] += 1
                elif self.score[1] == self.first_to[1]:
                    self.score = [0, 0]
                    self.games[1] += 1
                return image_data, reward

        def move_ball(self):
                """
                Move the ball in game state
                it calculates boundaries and it clips
                the ball positioning when it is overlapping
                with walls or paddles

                return rewards when right player makes contact with the ball
                and when ball leaves the game screen on the left side
                """
                reward = 0.0

                # get ball trajectory
                prev_x, prev_y = self.ballx, self.bally
                next_x, next_y = self.ballx + self.ball_speed_x, self.bally + self.ball_speed_y
                ball_trajectory = LineString([(prev_x, prev_y), (next_x, next_y)])

                # get possible collision lines
                upper_wall = LineString([(0, 0),
                                         (SCREEN_WIDTH, 0)])
                bottom_wall = LineString([(0, SCREEN_HEIGHT - BALL_SIZE),
                                          (SCREEN_WIDTH, SCREEN_HEIGHT - BALL_SIZE)])
                left_paddle = LineString([(self.cpux + PADDLE_WIDTH, self.cpuy - BALL_SIZE),
                                          (self.cpux + PADDLE_WIDTH, self.cpuy + PADDLE_HEIGHT)])
                right_paddle = LineString([(self.playerx - BALL_SIZE, self.playery - BALL_SIZE),
                                           (self.playerx - BALL_SIZE, self.playery + PADDLE_HEIGHT)])

                # chop ball trajectory when colliding
                if ball_trajectory.intersects(upper_wall):
                        self.ball_speed_y *= -1
                        upper = ball_trajectory.intersection(upper_wall)
                        self.ballx, self.bally = upper.x, upper.y + 1
                elif ball_trajectory.intersects(bottom_wall):
                        self.ball_speed_y *= -1
                        bottom = ball_trajectory.intersection(bottom_wall)
                        self.ballx, self.bally = bottom.x, bottom.y - 1
                elif ball_trajectory.intersects(left_paddle):
                        left = ball_trajectory.intersection(left_paddle)
                        contact_point = left.y - left_paddle.xy[1][0]
                        if contact_point < PADDLE_UPPER_SECTION or \
                           contact_point > PADDLE_BOTTOM_SECTION:
                                self.flip_and_spin_ball()
                        else:
                                self.flip_and_speed_ball()
                        self.ballx, self.bally = left.x + 1, left.y
                elif ball_trajectory.intersects(right_paddle):
                        reward += 0.1
                        right = ball_trajectory.intersection(right_paddle)
                        contact_point =  right.y - right_paddle.xy[1][0]
                        if contact_point < PADDLE_UPPER_SECTION or \
                           contact_point > PADDLE_BOTTOM_SECTION:
                                self.flip_and_spin_ball()
                        else:
                                self.flip_and_speed_ball()
                        self.ballx, self.bally = right.x - 1, right.y
                else:
                        self.ballx += self.ball_speed_x
                        self.bally += self.ball_speed_y

                return reward

        def draw_scores(self):
                """
                To be called when playing against
                human only so that numbers pixels don't
                interfere with learning
                """
                cpu_score = SCORE_FONT.render(str(self.score[0]), 1, (255, 255, 255))
                cpu_games = GAMES_FONT.render(str(self.games[0]), 1, (255, 255, 255))
                my_score = SCORE_FONT.render(str(self.score[1]), 1, (255, 255, 255))
                my_games = GAMES_FONT.render(str(self.games[1]), 1, (255, 255, 255))

                SCREEN.blit(cpu_score, (32, 16))
                SCREEN.blit(cpu_games, (32 - 4, 16))

                SCREEN.blit(my_score, (SCREEN_HEIGHT+92, 16))
                SCREEN.blit(my_games, (SCREEN_HEIGHT+92 - 4, 16))

        def complete_drawing(self):
                """
                Force the drawing of the screens
                """
                pygame.display.flip()
                if self.auto_draw: FPS_CLOCK.tick(QFPS)
                else: FPS_CLOCK.tick(FPS)

        def flip_and_spin_ball(self):
                """
                When ball makes contact with the upper
                or lower ends of either paddle, the ball
                will potentially randomly increase the y axis speed
                and be return with the same speed
                """
                self.ball_speed_x *= -1
                self.ball_speed_y *= random.randint(1000, 1200)/1000.

        def flip_and_speed_ball(self):
                """
                When the ball makes contact with the center
                of either paddle, it will return the ball with
                potentially an increase in the x axis speed
                y axis remains untouched
                """
                self.ball_speed_x *= -1
                self.ball_speed_x *= random.randint(1000, 1200)/1000.


def main(argv):
        """
        When called `python king_pong.py`
        a CPU is allocated to play against a human
        """
        game_state = GameState(auto_draw = False)

        # 2 game_states of 1 point
        game_state.first_to = [3, 2]
        game_state.top_speed = 5

        while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

                keys = pygame.key.get_pressed()
                a1 = keys[pygame.K_UP]
                a2 = 0 if a1 else keys[pygame.K_DOWN]
                a0 = 1 if not a1 and not a2 else 0
                image_data, reward = game_state.frame_step([a0, a1, a2])

                game_state.draw_scores()
                game_state.complete_drawing()

                if game_state.game_over():
                    exit(0)

if __name__ == "__main__":
    from sys import argv
    main(argv)

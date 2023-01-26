import pygame as pg
import random
import math
import gc
import numpy as np
import matplotlib.pyplot as plt
import time

pg.init()
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

window = pg.display.set_mode((800, 800))
myfont = pg.font.Font(pg.font.get_default_font(), 20)
robot_pic = pg.transform.scale(pg.image.load('images/chair.png'), (50, 50)).convert_alpha()

pg.display.set_caption('SLAM design')
chair_width = robot_pic.get_width()

class Table():
    def __init__(self, topx, topy, height, width, h):
        self.topx = topx
        self.topy = topy
        self.width = width
        self.h = h
        self.height = height
        self.random_angle = random.randint(0, 360)
        self.rec = pg.Rect((self.topx, self.topy), (self.width, self.h))


class Robot(pg.sprite.Sprite):
    IMG = robot_pic
    def __init__(self, posx, posy):
        pg.sprite.Sprite.__init__(self)
        self.pos_x = posx
        self.pos_y = posy
        self.img = self.IMG
        self.speed = 0.15
        self.rect = self.img.get_rect()
        self.direction = 0
        self.x = 0
        self.y = 50
        self.angle = 0
        self.dir = (0, 0)
    
    def rotate_img(self):
        mx, my = pg.mouse.get_pos()
        self.dir = (mx - self.pos_x, my - self.pos_y)
        length = math.hypot(*self.dir)
        if length == 0.0:
            self.dir = (0, -1)
        else:
            self.dir = (self.dir[0]/length, self.dir[1]/length)
        self.angle = math.degrees(math.atan2(-self.dir[1], self.dir[0]))
        rotated_img = pg.transform.rotate(self.img, self.angle-90)
        rotated_rect = rotated_img.get_rect(center=(self.pos_x, self.pos_y))
        window.blit(rotated_img, rotated_rect)
    
    def values_close(self, valOne, valTwo):
        dif = math.sqrt((valOne - valTwo) * (valOne - valTwo))
        return dif < 0.05


class Game():
    def __init__(self, tables):
        self.tables = tables
        self.bullets = []
        self.detected_table = False
        self.distance_security = 150
        self.min_height = 80
        self.detected = False

    def draw_tables(self):
        colors = [(255, 170, 15), (43, 100, 253), (43, 255, 100), (155, 100, 190)]
        count = 0
        for item in self.tables:
            pg.draw.rect(window, colors[count], item.rec)   
            count += 1

    def draw_game(self):
        angles = [x for x in range(-60, 270, 10)]
        robot.rotate_img()
        if len(self.bullets) < 20:
            self.bullets.append(Ball(robot.pos_x, robot.pos_y-robot.img.get_width()/2, angles[random.randint(0, len(angles)-1)]))
            
        for bullet in self.bullets:
            bullet.update()  
        remove_bullets = []
        for i in range(len(self.bullets)): 
            if self.bullets[i] in self.bullets and self.get_distance(self.bullets[i]) > self.distance_security:
                remove_bullets.append(self.bullets[i])
            else:
                self.bullets[i].draw_bullet()
            bullet_rect = self.bullets[i-1].rect()
            for item in self.tables:
                if pg.Rect.colliderect(item.rec, bullet_rect) and not self.detected:
                    print("Actual poses:", kalman.actual_poses)
                    print("Commands:", kalman.u)
                    print("Observations:", kalman.observation)
                    print("Mu robot:", kalman.mu[0, :])
                    print("Mu landmark:", kalman.mu[1, :])
                    print("sigma:", kalman.Sigma)
                    kalman.plot_graph()
                    self.detected = True
        for item in remove_bullets:
            if item in self.bullets:
                self.bullets.remove(item)
                gc.collect()
        rotated_img = pg.transform.rotate(robot.img, robot.angle-90)
        rotated_rect = rotated_img.get_rect(center=(robot.pos_x, robot.pos_y))
        window.blit(rotated_img, rotated_rect)
        for bullet in self.bullets:
            if self.get_distance(bullet) >= self.distance_security + 10:
                self.detected_table = False
        self.draw_tables()            
        pg.display.update()
        window.fill("white")
    
    def bounce(self):
        if robot.direction == 1:
            robot.pos_y += 1
        else:
            robot.pos_y -= 1
    
    def get_distance(self, bullet):
        dis = math.sqrt((robot.pos_x - bullet.pos_x)**2 + (robot.pos_y - bullet.pos_y) ** 2)
        return dis
    
    def intersect_table_chair(self, bullet, table):
        b_pos = bullet.center
        t_pos = table.rec.center
        dist = math.sqrt((t_pos[0] - b_pos[0])**2 + (t_pos[1] - b_pos[1]) ** 2)
        if dist < table.width/2:
            return True
        return False


class Kalman_Filter(object):
    def __init__(self):
        self.actual_poses = [0,0    ]
        self.predicted_pos = [0]
        self.observation = [0]
        self.sigma_seq = 0
        self.sigma_r = 1
        self.sigma_q = 0.5
        self.u = [0]
        self.R = np.array([[self.sigma_r**2, 0], [0, 0]])
        self.Q = self.sigma_q**2
        self.mu = np.zeros((2, 2))
        self.Sigma = [np.array([[0, 0], [0, 1e4]]), np.array([[0, 0], [0, 1e4]])]
        self.a = 1
        self.b = 1
        self.c = -1
        self.A = np.array([[self.a, 0], [0, 1]])
        self.B = np.array([self.b, 0])
        self.C = np.array([self.c, 1])
        self.landmarkSeen = 1
        pass

    def plot_graph(self):
        x = [i for i in range(len(kalman.mu[0,:]))]
        x2 = [[i for i in range(len(kalman.actual_poses))]]
        plt.scatter(x2, kalman.actual_poses, facecolors='none', edgecolors='r')
        plt.scatter(x, kalman.mu[0,:], facecolors='none', edgecolors='b')
        plt.scatter(x, kalman.mu[1,:], marker="^", facecolors='none', edgecolors='black')
        plt.show()
        pass

    
class Ball():
    def __init__(self, posx, posy, given_angle):
        self.pos_x = posx
        self.pos_y = posy
        self.given_angle = given_angle
        self.speed = 0.1
        mx, my = pg.mouse.get_pos()
        self.dir = (mx - self.pos_x + self.given_angle, my - self.pos_y + self.given_angle)
        length = math.hypot(*self.dir)
        if length == 0.0:
            self.dir = (0, -1)
        else:
            self.dir = (self.dir[0]/length, self.dir[1]/length)
        angle = math.degrees(math.atan2(-self.dir[1], self.dir[0]))
        self.bullet = pg.Surface((2, 2)).convert_alpha()
        self.bullet.fill((150, 150, 150))
        self.bullet = pg.transform.rotate(self.bullet, angle)        
    
    def update(self):
        self.pos_x += self.dir[0] * self.speed 
        self.pos_y += self.dir[1] * self.speed
    
    def draw_bullet(self):
        bullet_rect = self.bullet.get_rect(center=(self.pos_x, self.pos_y))
        window.blit(self.bullet, bullet_rect)
    
    def rect(self):
        bullet_rect = self.bullet.get_rect(center=(self.pos_x, self.pos_y))
        return bullet_rect


tables = []
# tables_rect = [(100, 400, 40), (800, 800, 85), (600, 500, 60), (400, 100, 90)]
# tables_rect = [(600, 450, 90)]
tables_rect = [(100, 100, 90)]
for item in tables_rect:
    tables.append(Table(item[0], item[1], item[2], 110, 70))

robot = Robot(150, 500)
run = True
kalman = Kalman_Filter()
game = Game(tables)
is_colliding = False
times = []
add_pose = True

while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
    force_x = math.cos(math.radians(robot.angle))
    force_y = math.sin(math.radians(robot.angle))
    keys = pg.key.get_pressed()
    for item in tables:
        chair_rect = robot.img.get_rect(topleft=(robot.pos_x - (robot.img.get_width()/2), robot.pos_y - (robot.img.get_height()/2)))
        collide = pg.Rect.colliderect(chair_rect,item.rec)
        if collide and item.height < game.min_height:
            is_colliding = True
            game.bounce()
    if keys[pg.K_z] and not is_colliding:
        start_time = time.time() 
        robot.pos_x += robot.speed * force_x
        robot.pos_y -= robot.speed * force_y
        robot.direction = 1
        times.append(start_time)
    if keys[pg.K_u]:
        if len(times) != 0:
            command = (times[-1] - times[0])*robot.speed
            kalman.u.append(round(command,4))
            times = []

    if keys[pg.K_s] and not is_colliding:
        robot.pos_x -= robot.speed * force_x
        robot.pos_y += robot.speed * force_y
        robot.direction = -1
    if is_colliding:
        is_colliding = False
    # Store the pos of the robot when pressed t
    if keys[pg.K_t]:
        ac_pos = kalman.a * kalman.actual_poses[-2] + kalman.b*kalman.u[-1] + kalman.sigma_r*random.uniform(0,1)
        ac_obs = kalman.c*ac_pos + game.distance_security + kalman.sigma_q*random.uniform(0,1)
        ac_mu = kalman.A*np.array(kalman.mu)[:,-2] + kalman.B*np.array(kalman.u)[-1]
        ac_sigma = kalman.A*kalman.Sigma[-2]*kalman.A.T + kalman.R
        if kalman.observation[-1] <= 149 & kalman.landmarkSeen == 1:
            kalman.mu[1, -1] = kalman.mu[0,-1] + kalman.observation[-1]
            kalman.landmarkSeen = 0
        if kalman.observation[-1] > 149:
            K = [0,0]
        else:
            K = kalman.Sigma[-1]*kalman.C.T*np.linalg.inv(kalman.c*kalman.Sigma[-1]*kalman.C.T+kalman.sigma_q)
        if add_pose:
            kalman.actual_poses.append(round(ac_pos,4))
            kalman.observation.append(round(ac_obs,4))
            kalman.mu = np.append(kalman.mu, ac_mu, axis=1)
            kalman.Sigma.append(ac_sigma)
            value = K*(kalman.observation[-1]-kalman.C*np.array(kalman.mu)[:,-1])
            np.array(kalman.mu)[:, -1] += [value[0][1], value[1][1]] if type(value[0]) == list or type(value[0]) == np.ndarray else [value[0], value[1]]
            np.array(kalman.Sigma)[-1] += (np.identity(2) - K*kalman.C)*kalman.Sigma[-1]
            add_pose = False
    if keys[pg.K_r]:
        add_pose = True
    game.draw_game() 

pg.quit()

# def uppercase_decorator(function):
#     def wrapper():
#         func = function()
#         make_uppercase = func.upper()
#         return make_uppercase

#     return wrapper

# @uppercase_decorator
# def say_hi():
#     return "hello world"

# print(say_hi())



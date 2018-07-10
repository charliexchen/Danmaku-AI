import math
import numpy as np
from functools import reduce


def sqdist(p1, p2):
    disvec = [p1[i] - p2[i] for i in range(2)]
    return sum([i ** 2 for i in disvec])


def col(p1, p2, r1, r2):
    return sqdist(p1, p2) < (r1 + r2) ** 2


def outofbound(pos, boundary, rad=0):
    if pos[0] < -rad or pos[0] > boundary[0] + rad or pos[1] < -rad or pos[1] > boundary[1] + rad:
        return True
    return False


class bullet():
    def __init__(self, vel, pos, rad):
        self.rad = rad
        self.vel = vel
        self.pos = pos

    def update(self, boundary):
        # boundary is 2d array of x, y coordinates
        self.pos = [self.pos[i] + self.vel[i] for i in range(2)]
        if outofbound(self.pos, boundary, self.rad):
            return False
        return True


class sensor():
    def __init__(self, pos, relpos, rad):
        self.relpos = relpos
        self.rad = rad
        self.pos = tuple(sum(t) for t in zip(self.relpos, pos))

    def sense(self, pos, bullets, boundary):
        self.pos = tuple(sum(t) for t in zip(self.relpos, pos))
        # work out absolute position
        if outofbound(self.pos, boundary):
            return 1
            # return 1 if outside the bounding box
        else:
            for bul in bullets:
                if col(self.pos, bul.pos, self.rad, bul.rad):
                    return 1
        return 0

    def mutate(self, scale):
        self.relpos = tuple(sum(t) for t in zip(self.relpos, np.random.normal(0, scale, 2)))


class ship():
    def __init__(self, maxvel, initpos, rad, sensorpos):
        self.maxvel = maxvel
        self.pos = initpos
        if type(sensorpos) != int:
            self.sensors = [sensor(initpos, i[:2], i[2]) for i in sensorpos]
        else:
            self.sensors = sensorpos
            self.highlightedpos=[(0,0)]*sensorpos
        self.rad = rad

    def move(self, vel):
        # inputvel = math.sqrt(sum([i ** 2 for i in vel]))
        # if inputvel > self.maxvel:
        #    vel = [i * self.maxvel / inputvel for i in vel]
        vel = [i * self.maxvel for i in vel]
        self.pos = [self.pos[i] + vel[i] for i in range(2)]

    def sense(self, bullets, boundary):
        output = []
        for sensor in self.sensors:
            output.append(sensor.sense(self.pos, bullets, boundary))
        return output

    def mutate_sensors(self, scale):
        for sensor in self.sensors:
            sensor.mutate(scale)


class environ():
    def __init__(self, hyperparams, boundary=[200, 200], bulletcap=100, shipinit=[100, 180], cd=5, spawn=[100, 10]):
        self.sensors, self.controller = hyperparams
        self.fighter = ship(6, shipinit, 1, self.sensors)
        self.shipinit = shipinit
        self.bullets = []
        self.cd = cd
        self.timer = cd
        self.boundary = boundary
        self.bulletcap = bulletcap
        self.spawn = spawn
        self.fitness = 0

    def reset(self):
        self.bullets = []
        self.fighter.pos = self.shipinit

    def update(self):
        # Spawn bullets
        self.timer -= 1
        if self.timer == 0:
            self.timer = self.cd
            if len(self.bullets) < self.bulletcap:
                spd = np.random.uniform(1.0, 3.5)
                angle = np.random.uniform(1.5 * 3.1416, 2.5 * 3.1416)
                v = [spd * np.sin(angle), spd * np.cos(angle)]
                self.bullets.append(bullet(v, self.spawn, 6))
        # Move the Bullets
        newbullets = []
        for i in range(len(self.bullets)):
            if self.bullets[i].update(self.boundary):
                newbullets.append(self.bullets[i])
        self.bullets = newbullets
        # Move the ship
        self.fighter.move(self.controller.evaluate(self.shipsensors()))

        # check collisions
        if outofbound(self.fighter.pos, self.boundary, self.fighter.rad):
            return True
        for bul in self.bullets:
            if col(self.fighter.pos, bul.pos, bul.rad, self.fighter.rad):
                return True
        return False

    def shipsensors(self):
        if type(self.fighter.sensors) == int:
            bulletpos = [bullet.pos for bullet in self.bullets]
            list.sort(bulletpos, key=lambda p: sqdist(p, self.fighter.pos))
            if self.fighter.sensors > len(self.bullets):
                bulletpos += [self.spawn] * (self.fighter.sensors - len(self.bullets))
            else:
                bulletpos = bulletpos[:self.fighter.sensors]
            self.fighter.highlightedpos=bulletpos
            closestdata = [[(bpos[i] - self.fighter.pos[i]) / self.boundary[i] for i in range(2)] for bpos in bulletpos]
            return list(reduce((lambda x, y: x + y), closestdata)) + [(self.fighter.pos[i] / self.boundary[i]) - 0.5 for
                                                                      i in range(2)]
        else:
            return self.fighter.sense(self.bullets, self.boundary) + [(self.fighter.pos[i] / self.boundary[i]) - 0.5 for
                                                                      i in range(2)]

    def eval_fitness(self, maxtime):
        self.reset()
        for i in range(maxtime):
            if self.update():
                self.fitness = i
                return i
        self.fitness = maxtime
        return maxtime

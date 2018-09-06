import math, copy
import numpy as np
from functools import reduce



def sqdist(p1, p2):
    disvec = [p1[i] - p2[i] for i in range(2)]
    return sum([i ** 2 for i in disvec])

def dot(v1, v2):
    try:
        assert len(v1)==len(v2)
        return sum([v1[i]+v2[i] for i in range( len(v1))])
    except AssertionError:
        print("Error -- dot product has different lengths {} and {}".format(len(v1), len(v2)))

def angle(p1, p2):
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    return math.atan2(dy, dx)

def col(p1, p2, r1, r2):
    return sqdist(p1, p2) < (r1 + r2) ** 2

def outofbound(pos, boundary, rad=0):
    if pos[0] < -rad or pos[0] > boundary[0] + rad or pos[1] < -rad or pos[1] > boundary[1] + rad:
        return True
    return False


class bullet:
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


class point_sensor:
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

class line_sensor:
    def __init__(self,pos, dir, dist=False):
        self.pos = pos
        self.dir = dir
        #direction of the line sensor is stored as a vector to simply calculations
        self.dist=dist

    def sense(self, pos, bullets, boundary):
        pass

class ship:
    def __init__(self, maxvel, initpos, rad, sensorpos, sensors= {"point":[(0, -5)], "ray":[], "prox":0, "pos":False}):
        #self.sensors = sensors
        #sets cap on speed
        self.maxvel = maxvel
        #initialises position
        self.pos = initpos
        #determines the type of sensors
        if type(sensorpos) != int:
            self.sensors = [point_sensor(initpos, i[:2], i[2]) for i in sensorpos]
        else:
            self.sensors = sensorpos
            self.highlightedpos = [(0, 0)] * sensorpos
        #size of hitbox
        self.rad = rad

    def move(self, vel):
        #moves plane by amount
        vel = [i * self.maxvel for i in vel]
        self.pos = [self.pos[i] + vel[i] for i in range(2)]

    def sense(self, bullets, boundary):
        #returns an array contains the information from the sensors
        output = []

        for sensor in self.sensors:
            output.append(sensor.sense(self.pos, bullets, boundary))
        return output

    def mutate_sensors(self, scale):
        for sensor in self.sensors:
            sensor.mutate(scale)

class environ:
    def __init__(self, hyperparams, boundary=[200, 200], bullet_cap=100, shipinit=[100, 100],
                 spawn=[100, 10], bullet_types={"aimed": 15, "spiral": 1, "random": 1}):
        # some parameters for the environment
        self.boundary = boundary
        self.bullet_cap = bullet_cap
        self.spawn = spawn

        # This is the hyperparameters for the environment -- the sensor values of ths ship and the neural network behind it
        self.sensors, self.controller = hyperparams
        # Initialise the ship
        self.fighter = ship(6, shipinit, 1, self.sensors)
        self.shipinit = shipinit

        # These dictinaries wstore the cooldowns for each bullet spawner
        self.bullets = []
        self.bullet_cooldowns = bullet_types
        self.bullet_counter = copy.deepcopy(bullet_types)

        # Define additional variables as needed for bullet types
        if "spiral" in bullet_types:
            self.dir = 0

        # Some values of the environment for reference
        self.fitness = 0
        self.deaths = 0

    def reset(self):
        #rests the state of the environment
        #clear the bullets
        self.bullets = []
        #reset ship position
        self.fighter.pos = self.shipinit

    def spawn_bullets(self):
        #This function is responsible for spawning the danmaku patterns
        for bullet_type in self.bullet_cooldowns:
            # decrement counters
            self.bullet_counter[bullet_type] -= 1
            #spawn bullets if countdown due
            if self.bullet_counter[bullet_type] < 0:
                self.bullet_counter[bullet_type] = self.bullet_cooldowns[bullet_type]
                if len(self.bullets) < self.bullet_cap:

                    if bullet_type == "aimed":
                        #aims bullet directly at ship -- forces agent to dodge or die
                        v = [(self.fighter.pos[0] - self.spawn[0]) / 20, (self.fighter.pos[1] - self.spawn[1]) / 20]
                        self.bullets.append(bullet(v, self.spawn, 10))

                    elif bullet_type == "spiral":
                        #predictable spiral pattern
                        self.dir += 80
                        while self.dir > 2.5 * 3.1416:
                            self.dir -= 3.1416
                        while self.dir < 1.5 * 3.1416:
                            self.dir += 3.1416
                        spd = 3.0
                        angle = self.dir
                        v = [spd * np.sin(angle), spd * np.cos(angle)]
                        self.bullets.append(bullet(v, self.spawn, 6))

                    elif bullet_type == "random":
                        #random spewing of bullets aimed at the ship
                        spd = np.random.uniform(2.0, 5)
                        angle = np.random.uniform(1.5 * 3.1416, 2.5 * 3.1416)
                        v = [spd * np.sin(angle), spd * np.cos(angle)]
                        self.bullets.append(bullet(v, self.spawn, 6))

    def update(self):
        # One step in the process, returns True if the ship dies
        # Spawn bullets
        self.spawn_bullets()

        # Move the Bullets
        newbullets = []
        for i in range(len(self.bullets)):
            if self.bullets[i].update(self.boundary):
                newbullets.append(self.bullets[i])
        self.bullets = newbullets

        # Move the ship according to sensor data
        self.fighter.move(self.controller.evaluate(self.shipsensors()))

        # check collisions
        # ship and boundary of screen
        if outofbound(self.fighter.pos, self.boundary, self.fighter.rad):
            self.deaths += 1
            return True
        # ship and the bullets
        for bul in self.bullets:
            if col(self.fighter.pos, bul.pos, bul.rad, self.fighter.rad):
                self.deaths += 1
                return True
        #return False if no collision occurs
        return False

    def shipsensors(self, radial=True):
        #returns sensor values

        if type(self.fighter.sensors) == int:
            # Try to detect closest bullets
            boundarypos = [[self.fighter.pos[0], 0], [self.fighter.pos[0], self.boundary[1]], [0, self.fighter.pos[1]],
                           [self.boundary[0], self.fighter.pos[1]]]
            bulletpos = [bullet.pos for bullet in self.bullets]
            bulletpos = bulletpos + boundarypos
            list.sort(bulletpos, key=lambda p: sqdist(p, self.fighter.pos))
            if self.fighter.sensors > len(bulletpos):
                bulletpos += [self.spawn] * (self.fighter.sensors - len(bulletpos))
            else:
                bulletpos = bulletpos[:self.fighter.sensors]
            self.fighter.highlightedpos = bulletpos
            if radial:
                closestdata = [[math.log(sqdist(bpos, self.fighter.pos)), angle(bpos, self.fighter.pos)] for
                               bpos in bulletpos]
            else:
                closestdata = [[(bpos[i] - self.fighter.pos[i]) / self.boundary[i] for i in range(2)] for bpos in
                               bulletpos]
            output = list(reduce((lambda x, y: x + y), closestdata))
            return output
        else:
            # Return a binary array indicating which sensor has been triggered
            return self.fighter.sense(self.bullets, self.boundary) + [(self.fighter.pos[i] / self.boundary[i]) - 0.5 for
                                                                      i in range(2)]

    def eval_fitness(self, maxtime):
        # Runs an episode where the environment spawns bullets until the ship is hit
        fitness = 0
        self.reset()
        for i in range(maxtime):
            # Rewards the agent for every frame for which it remains alive
            # Reward higher near the centre of the screen.
            fitness += 1 - abs((2 * self.fighter.pos[0] / self.boundary[0])- 1)
            if self.update():
                self.fitness = fitness
                return fitness
        self.fitness = fitness
        return fitness

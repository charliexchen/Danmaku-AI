import copy
import numpy as np
from functools import reduce
from Game.enums import *
from Game.utils import *


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


class laser:
    def __init__(self, pos, v, r=5):
        self.rad = r
        self.pos = pos
        self.v = v
        self.dir = angle((0, 0), v)

    def update(self, boundary):
        self.pos = [self.pos[i] + self.v[i] for i in range(2)]
        if outofbound(self.pos, boundary, self.rad):
            return False
        return True


class point_sensor:
    def __init__(self, pos, relpos, rad):
        self.relpos = relpos
        self.rad = rad
        self.pos = tuple(sum(t) for t in zip(self.relpos, pos))
        self.on = 0

    def sense(self, pos, bullets, boundary):
        self.pos = tuple(sum(t) for t in zip(self.relpos, pos))
        # work out absolute position
        if outofbound(self.pos, boundary):
            self.on = 1
            return self.on
            # return 1 if outside the bounding box
        else:
            for bul in bullets:
                if col(self.pos, bul.pos, self.rad, bul.rad):
                    self.on = 1
                    return self.on
        self.on = 0
        return self.on

    def mutate(self, scale):
        self.relpos = tuple(
            sum(t) for t in zip(self.relpos, np.random.normal(0, scale, 2))
        )


class line_sensor:
    def __init__(self, dir, boundary, log=True):
        self.dir = sub_angle(dir, 0)
        # direction of the line sensor is stored as a vector to simply calculations
        self.boundary = boundary
        self.max_output = sqdist((0, 0), boundary)
        self.dist = self.max_output
        self.log = log

    def sense(self, pos, bullets):
        output = self.max_output
        if math.cos(self.dir) != 0:
            if abs(self.dir) < math.pi / 2:
                output = min(
                    output, abs((pos[1] - self.boundary[1]) / math.cos(self.dir))
                )
            else:
                output = min(output, abs(pos[1] / math.cos(self.dir)))
        if math.sin(self.dir) != 0:
            if self.dir > 0:
                output = min(
                    output, abs((pos[0] - self.boundary[0]) / math.sin(self.dir))
                )
            else:
                output = min(output, abs(pos[0] / math.sin(self.dir)))

        for bullet in bullets:
            b_dir = angle(pos, bullet.pos)
            dir_diff = abs(sub_angle(b_dir, self.dir))
            d = dist(pos, bullet.pos)
            if bullet.rad / d <= 1:
                theta = math.asin(bullet.rad / d)
                if abs(dir_diff) < abs(theta):
                    # The bullet is in the sensor's line of sight
                    # However, the distance to the hitbox is not necessarily the distance to the bullet
                    # Here we calculate the difference
                    opposite = math.sin(dir_diff) * d
                    adjacent = math.cos(dir_diff) * d
                    if bullet.rad ** 2 - opposite ** 2 < 0:
                        difference = 0
                    else:
                        difference = math.sqrt(bullet.rad ** 2 - opposite ** 2)
                    output = min(output, adjacent - difference)
            else:
                return 0
        self.dist = output
        if self.log:
            return math.log(output, 5)
        return output


class ship:
    def __init__(
        self,
        max_vel,
        init_pos,
        rad,
        boundary,
        env,
        sensors={"point": [(0, -5)], "line": 8, "prox": 0, "pos": False},
        cooldown=6,
        focusing=True,
    ):
        self.sensors = sensors
        self.input_len = 0
        if "point" in sensors:
            self.input_len += len(sensors["point"])
        if "prox" in sensors:
            self.input_len += 2 * sensors["prox"]
        if "loc" in sensors:
            self.input_len += 2
        if "line" in sensors:
            self.input_len += sensors["line"]
        # sets cap on speed
        self.max_vel = max_vel

        # initialises position
        self.pos = init_pos
        self.max_cooldown = cooldown
        self.cooldown = cooldown
        self.focusing = focusing
        if focusing:
            self.focus = True
            self.foc_vel = max_vel / 3
        self.env = env

        # determines the type of sensors
        if sensorType.POINT in self.sensors:
            self.point_sensors = [
                point_sensor(init_pos, i[:2], i[2])
                for i in self.sensors[sensorType.POINT]
            ]

        if sensorType.PROXIMITY in self.sensors:
            self.prox_sensors = self.sensors[sensorType.PROXIMITY]
            self.highlightedpos = [(0, 0)] * self.prox_sensors

        if sensorType.LINE in self.sensors:
            count = self.sensors[sensorType.LINE]
            self.line_sensors = [
                line_sensor(2 * math.pi * i / count, boundary) for i in range(count)
            ]

        # size of hitbox
        self.rad = rad

    def move(self, input):
        if self.focusing:
            self.focus = input[2] < 0
            if self.focus:
                self.cooldown -= 3
                if not self.cooldown > 0:
                    self.cooldown = self.max_cooldown
                    self.env.spawn_laser(self.pos, self.focus)
            else:
                self.cooldown -= 1
                if not self.cooldown > 0:
                    self.cooldown = self.max_cooldown
                    self.env.spawn_laser(self.pos, self.focus)

            move = [i * self.max_vel for i in input[:2]]
            if self.focus:
                move = [v if v < self.foc_vel else self.foc_vel for v in move]
                move = [v if v > -self.foc_vel else -self.foc_vel for v in move]
            self.pos = [self.pos[i] + move[i] for i in range(2)]
        else:
            self.cooldown -= 1
            if not self.cooldown > 0:
                self.cooldown = self.max_cooldown
                self.env.spawn_laser(self.pos, False)
            input = [i * self.max_vel for i in input]
            self.pos = [self.pos[i] + input[i] for i in range(2)]

    def sense(self, bullets, boundary):
        # returns an array contains the information from the sensors
        output = []
        if "line" in self.sensors:
            for sensor in self.line_sensors:
                output.append(sensor.sense(self.pos, bullets))
        if "point" in self.sensors:
            for sensor in self.point_sensors:
                output.append(sensor.sense(self.pos, bullets, boundary))
        return output

    def mutate_sensors(self, scale):
        if "point" in self.sensors:
            for sensor in self.point_sensors:
                sensor.mutate(scale)


class environ:
    def __init__(
        self,
        hyperparams,
        boundary=[200, 200],
        bullet_cap=1000,
        shipinit=[100, 100],
        enemy_spawn=[100, 10],
        bullet_types={bulletType.AIMED: 15, bulletType.SPIRAL: 1, bulletType.RANDOM: 1},
    ):
        # some parameters for the environment
        self.boundary = boundary
        self.bullet_cap = bullet_cap
        self.spawn = enemy_spawn
        self.spawn_speed = 1
        self.damage = 0

        # This is the hyperparameters for the environment -- the sensor values of ths ship and the neural network behind it
        self.sensors, self.controller = hyperparams

        # Initialise the ship
        self.fighter = ship(8, shipinit, 0, boundary, self, self.sensors)
        self.shipinit = shipinit

        # These dictinaries store the cooldowns for each bullet spawner
        self.bullets = []
        self.bullet_cooldowns = bullet_types
        self.bullet_counter = copy.deepcopy(bullet_types)

        # Define additional variables as needed for bullet types
        if "spiral" in bullet_types:
            self.dir = 0

        # Some values of the environment for reference
        self.fitness = 0
        self.deaths = 0
        self.move_dir = [0, 0]
        self.lasers = []

    def reset(self):
        # rests the state of the environment
        # clear the bullets
        self.bullets = []
        self.lasers = []
        self.damage = 0
        # reset ship position
        self.fighter.pos = self.shipinit
        self.controller.previous = [0 for i in self.controller.previous]

    def spawn_bullets(self):
        # This function is responsible for spawning the bullet patterns
        for bullet_type in self.bullet_cooldowns:
            # decrement counters
            self.bullet_counter[bullet_type] -= 1
            # spawn bullets if countdown due
            if self.bullet_counter[bullet_type] < 0:
                self.bullet_counter[bullet_type] = self.bullet_cooldowns[bullet_type]
                if len(self.bullets) < self.bullet_cap:

                    if bullet_type == "aimed":
                        # aims bullet directly at ship -- forces agent to dodge or die
                        v = [
                            (self.fighter.pos[0] - self.spawn[0]) / 20,
                            (self.fighter.pos[1] - self.spawn[1]) / 20,
                        ]

                        self.bullets.append(bullet(v, self.spawn, 10))

                    elif bullet_type == "spiral":
                        # predictable spiral pattern
                        for i in range(1):
                            self.dir += 80
                            while self.dir > 2.5 * math.pi:
                                self.dir -= math.pi
                            while self.dir < 1.5 * math.pi:
                                self.dir += math.pi
                            spd = 3.0
                            angle = self.dir
                            v = [spd * np.sin(angle), spd * np.cos(angle)]
                            self.bullets.append(bullet(v, self.spawn, 7))

                    elif bullet_type == "random":
                        # random spewing of bullets aimed at the ship
                        for i in range(1):
                            spd = np.random.uniform(2.0, 5)
                            angle = np.random.uniform(1.5 * math.pi, 2.5 * math.pi)
                            v = [spd * np.sin(angle), spd * np.cos(angle)]
                            self.bullets.append(bullet(v, self.spawn, 6))

    def spawn_laser(self, pos, focused=True):
        unfoc_patter = [
            [[0, -5], [0, -10]],
            [[0, -5], [2, -9]],
            [[0, -5], [-2, -9]],
            [[0, -5], [4, -8]],
            [[0, -5], [-4, -8]],
        ]
        foc_patter = [
            [[5, -5], [0, -15]],
            [[-5, -5], [0, -15]],
            [[-5, -5], [2, -15]],
            [[5, -5], [-2, -15]],
        ]
        if focused:
            for shot in foc_patter:
                self.lasers.append(
                    laser([shot[0][i] + pos[i] for i in range(2)], shot[1])
                )
        else:
            for shot in unfoc_patter:
                self.lasers.append(
                    laser([shot[0][i] + pos[i] for i in range(2)], shot[1])
                )

    def update(self):
        self.spawn[0] += self.spawn_speed
        if self.spawn[0] > 170:
            self.spawn_speed = -1
        if self.spawn[0] < 30:
            self.spawn_speed = 1
        # One step in the process, returns True if the ship dies
        # Spawn bullets
        self.spawn_bullets()

        # Move the Bullets
        new_bullets = []
        for i in range(len(self.bullets)):
            if self.bullets[i].update(self.boundary):
                new_bullets.append(self.bullets[i])
        self.bullets = new_bullets

        # Move the lasers
        new_lasers = []
        for i in range(len(self.lasers)):
            if self.lasers[i].update(self.boundary):
                if col(self.lasers[i].pos, self.spawn, 20, self.lasers[i].rad):
                    self.damage += 1
                else:
                    new_lasers.append(self.lasers[i])
        self.lasers = new_lasers

        # Move the ship according to sensor data
        self.fighter.move(self.controller.activate(self.ship_sensors()))

        # Check collisions -- return True if the ship dies
        # Ship and boundary of screen
        if outofbound(self.fighter.pos, self.boundary, self.fighter.rad):
            self.deaths += 1
            return True
        # Ship and the bullets
        for bul in self.bullets:
            if col(self.fighter.pos, bul.pos, bul.rad, self.fighter.rad):
                self.deaths += 1
                return True
        # Return False if no collision occurs
        return False

    def ship_sensors(self, radial=True):
        # Returns sensor values
        output = []
        if (
            sensorType.POINT in self.fighter.sensors
            or sensorType.LINE in self.fighter.sensors
        ):
            # Return a binary array indicating which sensor has been triggered
            output += self.fighter.sense(self.bullets, self.boundary)
        if (
            sensorType.PROXIMITY in self.fighter.sensors
            and self.fighter.sensors[sensorType.PROXIMITY] > 0
        ):
            # Try to detect closest bullets
            boundary_pos = [
                [self.fighter.pos[0], 0],
                [self.fighter.pos[0], self.boundary[1]],
                [0, self.fighter.pos[1]],
                [self.boundary[0], self.fighter.pos[1]],
            ]
            bullet_pos = [bullet.pos for bullet in self.bullets]
            bullet_pos = bullet_pos + boundary_pos
            list.sort(bullet_pos, key=lambda p: sqdist(p, self.fighter.pos))
            if self.fighter.prox_sensors > len(bullet_pos):
                bullet_pos += [self.spawn] * (
                    self.fighter.prox_sensors - len(bullet_pos)
                )
            else:
                bullet_pos = bullet_pos[: self.sensors[sensorType.PROXIMITY]]
            self.fighter.highlightedpos = bullet_pos
            if radial:
                closestdata = [
                    [
                        math.log(sqdist(bpos, self.fighter.pos)),
                        angle(bpos, self.fighter.pos),
                    ]
                    for bpos in bullet_pos
                ]
            else:
                closestdata = [
                    [
                        (bpos[i] - self.fighter.pos[i]) / self.boundary[i]
                        for i in range(2)
                    ]
                    for bpos in bullet_pos
                ]
            output += list(reduce((lambda x, y: x + y), closestdata))

        if sensorType.LOCATION in self.fighter.sensors:
            output += [(self.fighter.pos[i] / self.boundary[i]) - 0.5 for i in range(2)]
        if self.fighter.focusing:
            output += [(self.spawn[i] / self.boundary[i]) - 0.5 for i in range(2)]
        return output

    def eval_fitness(self, max_time):
        # Runs an episode where the environment spawns bullets until the ship is hit
        fitness = 0
        self.reset()
        for i in range(max_time):
            # Rewards the agent for every frame for which it remains alive
            # Reward higher near the centre of the screen.
            fitness += 1 - abs((2 * self.fighter.pos[0] / self.boundary[0]) - 1)
            if self.update():
                self.fitness = fitness
                return fitness
        self.fitness = fitness
        return fitness

    def eval_dmg(self, max_time):
        # Runs an episode where the environment spawns bullets until the ship is hit
        self.reset()
        for i in range(max_time):
            # Rewards the agent for damage dealt to enemy ship
            if self.update():
                return self.damage
        return self.damage

import pygame
import pickle, math
from objects import environ, angle
from neural_net import dense_net, relu, sigmoid, tanh
import pdb, cProfile

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
DRED = (120, 0, 0)
GREEN = (0, 255, 0)
DGREEN = (0, 120, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
DMAGENTA = (120, 0, 120)


# function for rounding floats for display
def dispos(input):
    return [int(i) for i in input]


class gui:
    def __init__(self, boundary=[200, 200]):
        self.boundary = boundary

    def display_imported_generation(self, filename="generation450.p", loops=-1,
                                    bullet_types={"spiral": 1, "aimed:": 15}):
        trained_pop = pickle.load(open(filename, "rb"))
        # pdb.set_trace()
        print("Imported {}".format(filename))

        fittest_value = max(trained_pop["fitness"])
        fittest_index = trained_pop["fitness"].index(fittest_value)

        fittest_net = trained_pop["nets"][fittest_index]
        print("Fitness of best performer: {}".format(fittest_value))

        sensors = trained_pop["sensor_type"]
        if "prox" in sensors:
            print("{} proxmimity sensors extracted".format(sensors["prox"]))
        if "loc" in sensors:
            print("Location sensors extracted")
        if "pixel" in sensors:
            print("Pixels sensors extracted")

        self.display_net((sensors, fittest_net), loops, bullet_types)

    def display_net(self, hyperparams, loops=-1, bullet_types={"aimed:": 15, "random": 1, "spiral": 2},
                    displaySensors=set(["point", "line", "prox"])):

        pygame.init()

        screen = pygame.display.set_mode(self.boundary)

        # Set the title of the window
        pygame.display.set_caption('bullet dodge')

        clock = pygame.time.Clock()
        done = False
        env = environ(hyperparams, self.boundary, 100, [100, 100], [100, 10], bullet_types)

        #We need to display the ship one frame behind, since it moves after the sensors are activated
        prev_pos =[0,0]
        while not done:
            if loops == env.deaths:
                done = True
            if env.update():
                print("Hit!")
                env.reset()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True
            screen.fill(BLACK)
            for bullet in env.bullets:
                pygame.draw.circle(screen, WHITE, dispos(bullet.pos), bullet.rad)
            #update the sensors again, since we move the plane after the sensors are updated
            if displaySensors:
                if "line" in env.fighter.sensors and "line" in displaySensors:
                    for sensor in env.fighter.line_sensors:
                        detect_pos = [math.sin(sensor.dir) * sensor.dist + env.fighter.pos[0],
                                      math.cos(sensor.dir) * sensor.dist + env.fighter.pos[1]]
                        pygame.draw.line(screen, DMAGENTA, env.fighter.pos, detect_pos)
                if "point" in env.fighter.sensors and "point" in displaySensors:
                    for sensor in env.fighter.point_sensors:
                        if sensor.on:
                            pygame.draw.circle(screen, RED, dispos(sensor.pos), 3)
                        else:
                            pygame.draw.circle(screen, DGREEN, dispos(sensor.pos), 1)

                if "prox" in env.fighter.sensors and "prox" in displaySensors:
                    for incoming in env.fighter.highlightedpos:
                        pygame.draw.line(screen, DRED, dispos(incoming), dispos(env.fighter.pos))
                        if incoming[0] == 0:
                            pygame.draw.line(screen, RED, (1, 0), (1, self.boundary[1]))
                        elif incoming[0] == self.boundary[0]:
                            pygame.draw.line(screen, RED, (self.boundary[0] - 1, 0),
                                             (self.boundary[0] - 1, self.boundary[1]))
                        elif incoming[1] == 0:
                            pygame.draw.line(screen, RED, (0, 1), (self.boundary[0], 1))
                        elif incoming[1] == self.boundary[1]:
                            pygame.draw.line(screen, RED, (0, self.boundary[1] - 1),
                                             (self.boundary[0], self.boundary[1] - 1))
                        else:
                            pygame.draw.circle(screen, RED, dispos(incoming), 10, 1)
                p1 = dispos(prev_pos)[0]
                p2 = dispos(prev_pos)[1]
                pygame.draw.polygon(screen, WHITE, [[p1, p2 - 4], [p1 + 2, p2 + 3], [p1 - 2, p2 + 3]])
                pygame.draw.circle(screen, CYAN, dispos(prev_pos), env.fighter.rad + 1)

            else:
                p1 = dispos(prev_pos)[0]
                p2 = dispos(prev_pos)[1]
                pygame.draw.polygon(screen, WHITE, [[p1, p2 - 5], [p1 + 3, p2 + 4], [p1 - 3, p2 + 4]])
                pygame.draw.circle(screen, CYAN, dispos(prev_pos), env.fighter.rad)
            prev_pos = env.fighter.pos
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    sensors = {"point": [
        (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
        (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
        (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
        (0, -50, 1), (10, -20, 1), (-10, -20, 1), (-50, 0, 1), (50, 0, 1),
        (0, 15, 1), (10, -30, 1), (-10, -30, 1)],
        "prox": 3, "line": 12, "loc": True}

    input_len = 0
    if "point" in sensors:
        input_len += len(sensors["point"])
    if "prox" in sensors:
        input_len += 2 * sensors["prox"]
    if "line" in sensors:
        input_len += sensors["line"]
    if "loc" in sensors:
        input_len += 2
    net = dense_net(input_len, 10, relu, recursive=False)
    net.add_layer(10, tanh)
    # env = environ((8, net))
    # print("Profiling fitness evaluation times")
    # cProfile.run('env.eval_fitness(500)')
    GUI = gui()
    hyperparams = (sensors, net)
    GUI.display_imported_generation("saved_nets/AllSensors290.p", bullet_types={"spiral": 1, "random": 1})
    # GUI.display_net(hyperparams, bullet_types={"spiral": 1, "aimed:": 15})

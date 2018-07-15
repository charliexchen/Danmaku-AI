import pygame
import pickle
from Environ.objects import environ
from Environ.neural_net import dense_net, relu, sigmoid, tanh

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)


# function for rounding floats for display
def dispos(input):
    return [int(i) for i in input]


class gui:
    def __init__(self, boundary=[200, 200]):
        self.boundary = boundary

    def display_imported_generation(self, filename="generation450.p", loops=-1):
        trained_pop = pickle.load(open(filename, "rb"))
        print("Imported {}".format(filename))
        fittest = max(trained_pop, key=lambda x: x.fitness)
        print("Fitness of best performer: {}".format(fittest.fitness))
        net = fittest.controller

        if type(fittest.fighter.sensors) == int:
            sensortype = "Proximity"
            sensorpos = fittest.sensors
        else:
            sensortype = "Pixel"
            sensorpos = [tuple(list(sensor.relpos) + [1.0]) for sensor in fittest.fighter.sensors]
        print("Extracted sensor of type: {}".format(sensortype))

        self.display_net((sensorpos, fittest.controller), loops)

    def display_net(self, hyperparams, loops=-1, cd=10000, rand_cd=1, rand_bul=True):

        pygame.init()
        # Create an 800x600 sized screen

        screen = pygame.display.set_mode(self.boundary)

        # Set the title of the window
        pygame.display.set_caption('bullet dodge')

        clock = pygame.time.Clock()
        done = False
        env = environ(hyperparams, self.boundary, 100, [100, 100], cd, rand_cd, [100, 10], rand_bul)
        while not done:
            if loops == env.deaths:
                done = True
            if env.update():
                env.reset()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True
            screen.fill(BLACK)
            for bullet in env.bullets:
                pygame.draw.circle(screen, WHITE, dispos(bullet.pos), bullet.rad)
            pygame.draw.circle(screen, CYAN, dispos(env.fighter.pos), env.fighter.rad)
            activesensors = env.shipsensors()
            if type(env.fighter.sensors) != int:
                for i in range(len(env.fighter.sensors)):
                    if activesensors[i] == 0:
                        pygame.draw.circle(screen, GREEN, dispos(env.fighter.sensors[i].pos), 1)
                    else:
                        pygame.draw.circle(screen, RED, dispos(env.fighter.sensors[i].pos), 3)
            else:
                for incoming in env.fighter.highlightedpos:
                    pygame.draw.line(screen, RED, dispos(incoming), dispos(env.fighter.pos))
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
                # pygame.draw.circle(screen,, dispos(env.fighter.pos), env.fighter.rad)
            pygame.display.flip()
            clock.tick(15)
        pygame.quit()


sensorpos = [
    (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
    (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
    (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
    (0, -50, 1), (10, -20, 1), (-10, -20, 1), (-50, 0, 1), (50, 0, 1),
    (0, 15, 1), (10, -30, 1), (-10, -30, 1)]

net = dense_net(16, 10, relu, recursive=True)
net.add_layer(2, tanh)
env = environ((8, net), cd=10000)
GUI = gui()
hyperparams = (8, net)
GUI.display_net(hyperparams)

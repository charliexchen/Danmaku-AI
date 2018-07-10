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


pygame.init()
# Create an 800x600 sized screen
boundary = [200, 200]
screen = pygame.display.set_mode(boundary)

# Set the title of the window
pygame.display.set_caption('bullet dodge')

clock = pygame.time.Clock()
done = False

sensorpos1 = [
    (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
    (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
    (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
    (0, -50, 1), (10, -20, 1), (-10, -20, 1), (-50, 0, 1), (50, 0, 1),
    (0, 15, 1), (10, -30, 1), (-10, -30, 1)]

net = dense_net(10, 10, relu)
net.add_layer(2, tanh)

trained_pop = pickle.load(open("generation44.p", "rb"))
net = max(trained_pop, key = lambda x : x.fitness).controller
#sensorpos1 = [tuple(list(sensor.relpos)+[1.0]) for sensor in  max(trained_pop, key = lambda x : x.fitness).fighter.sensors]
#print (sensorpos1)

env = environ((4, net), cd=1)


#for i in range(5):
#    print(env.eval_fitness(10000))
while not done:
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
    if type(env.fighter.sensors)!=int:
        for i in range(len(env.fighter.sensors)):
            if activesensors[i] == 0:
                pygame.draw.circle(screen, GREEN, dispos(env.fighter.sensors[i].pos), 1)
            else:
                pygame.draw.circle(screen, RED, dispos(env.fighter.sensors[i].pos), 3)
    else:
        for incoming in env.fighter.highlightedpos:
            pygame.draw.circle(screen, RED, dispos(incoming), 10, 1)
            pygame.draw.line(screen, RED, dispos(incoming), dispos(env.fighter.pos))


        #pygame.draw.circle(screen,, dispos(env.fighter.pos), env.fighter.rad)
    pygame.display.flip()
    clock.tick(25)
pygame.quit()

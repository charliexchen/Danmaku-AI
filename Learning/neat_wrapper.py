"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import neat
from Game.objects import environ

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

# sensors = {"point": [
#    (0, -10, 1), (10, -10, 1), (-10, -10, 1), (-10, 0, 1), (10, 0, 1),
#    (0, -20, 1), (20, -20, 1), (-20, -20, 1), (-20, 0, 1), (20, 0, 1),
#    (0, -30, 1), (30, -30, 1), (-30, -30, 1), (-30, 0, 1), (30, 0, 1),
#    (0, -50, 1), (10, -20, 1), (-10, -20, 1), (-50, 0, 1), (50, 0, 1),
#    (0, 15, 1), (10, -30, 1), (-10, -30, 1), (0, -3, 1), (3, 0, 1), (-3, 0, 1)],
#    "prox": 5, "loc": True}
sensors = {"line": 16, "loc": True}


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = environ((sensors, net), bullet_types={"spiral": 1, "aimed": 10})
    return env.eval_fitness(1000)


if __name__ == "__main__":
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config-feedforward",
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    p.add_reporter(neat.Checkpointer(5, filename_prefix="saved_nets/neat-checkpoint-"))
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

    # Run until a solution is found.

    pe = neat.ParallelEvaluator(36, eval_genome)
    winner = p.run(pe.evaluate, 5000)

    # Display the winning genome.
    print("\nBest genome:\n{!s}".format(winner))

    # Show output of the most fit genome against training data.
    print("\nOutput:")
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = winner_net.activate(xi)
    #    print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

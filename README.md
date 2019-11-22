# DanmakuBot

Danmaku (弹幕) refers to these really hard arcade shoot 'em up games (like space invaders), where the entire screen gets covered in bullets, and the player has to weave between the bullets in order to survive.

Years ago, in high school (2012), I made such a game in GameMaker. Because it was such a difficult game, I attempted to implement a bot to autoplay it. This turned out to be quite hard, and the AI I wrote back then could only survive a couple of seconds before it gets stuck in a corner and died.

This is a second attempt at that problem using neural networks and evolution in 2018, before I started my first job.

As a challenge, I did not use deep learning libraries such as TensorFlow or PyTorch. Instead, everything is implemented with numpy (though as luck would have it, this can now be run on the gpu with JAX with minimal changes).


# Methodology

The game is implemented from scratch, and can be run without rendering to save CPU. Rendering can be done using pygame.

The optimisation algorithm is simple, truncation evolution -- create agents, take best performing agents, add noise to create new population, repeat.

The agent below is trained using an AWS instance with 36 cores for ~250 generations.


# What I learned

* Use AWS properly

* MultiProcessing in order to accelerate evolution

* Sometimes, even stupid algorithms can work really well if enough. In fact, I this was likely a more robust algorithm than, say policy gradient.




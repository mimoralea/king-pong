# King Pong
### Deep Reinforcement Learning Pong Player

![King Pong][king]

## Overview
In this repository, you have an agent that plays the game of pong. Make no mistake though, this is not a normal player. King (the agent) has learned to play the game of pong all by himself, by looking at the screen just like you would. Now, as you can imagine, there are a lot of cutting edge technologies being mixed into this project. First, we have Computer Vision to be able to receive the percepts from the screen. Next, we have Reinforcement Learning which is part of Machine Learning, but it is not classification, nor regression, or clustering.


### What is Reinforcement Learning?
[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning) is inspired by the study of animal behavior. In specific, how animals react to pain, reward signals through time. King wants to win, that's why he learns to do what he does.

![Reinforcement Learning][rl]

### What is Deep Learning?
The problem with Reinforcement Learning comes when the number of states in which the environment could present itself is too large. Think about it, if you see some pointy object approaching your hand, you might immediately protect it. Even if you have never been hurt by exactly the same object. This ability that you have to extrapolate states is "Deep Learning" in your brain. [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) is a part of Machine Learning that specializes on function approximation. In the case of King, we are using Deep Learning so percepts that have never been seen (think pointy object) still get treated with the same regard as similar percepts would (move your hand out of the way.)

![Deep Learning][dl]

### What is Deep Reinforcement Learning?
Deep Reinforcement Learning is the combination of these two techniques to make a stronger learning approach. An amazing blog post that helped me tremendously is called [Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/) by Tambet Matiisen.

![Deep Reinforcement Learning][drl]

## Installation

### Dependencies

The following dependencies have to be installed before running the code:

- [Pygame](http://www.pygame.org/wiki/GettingStarted)
- [TensorFlow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup)
- [OpenCV2](http://opencv.org/)
- [Shapely](https://pypi.python.org/pypi/Shapely)
- [Numpy](http://www.scipy.org/scipylib/download.html)

**NOTE**: Most of this packages are straight forward to install using `pip` or `conda` but "OpenCV" in particular could be challenging. I highly recommend using your OS package management system to get it installed.

For example in Fedora, the following command would do just fine:

```bash
sudo dnf install opencv-python
```

### Docker image

To make things a little easier, I have created a Docker image that will hopefully help with running the code. You can either `pull` the image I have pushed into Docker.io, or you can build it from scratch.

#### To pull execute:

```bash
docker pull mimoralea/king-pong:v1
```

This command will download the container with all of the dependencies installed.


#### To build execute:

**NOTE:** You don't need this command if you pull the container (above).

```bash
docker build -t mimoralea/king-pong:v1 .
```

This command can take a little while, but it will build a Fedora image and install all the depencies for you.

#### To get into the container:

```bash
docker run --privileged=true --rm \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/king-pong \
    -it mimoralea/king-pong:v1 /bin/bash
```

This command redirects the X11 session to your local computer. You need this so that Pygame can open the window in your local system. This command also maps the local current directory `.` containing the source code to the `/king` directory in the container. So please make sure that you are inside the project's root directory.

### Run the app:

If you used the Docker method you need to get into the `king-pong` directory first. `cd king-pong`. If you installed the pages locally, the `agent.py` file should be in your current directory. To run the application run from the project's root directory:

```bash
cd /king-pong
python agent.py -vv -g 1 -m 3
```

This will start a quick 3 points game.

## Commands available:

```
usage: agent.py [-h] [-v] [-vv] [-vvv] [-g NGAMES] [-m NMATCHES] [-t] [-c]

A Deep Reinforcement Learning agent that plays pong like a King.

optional arguments:
  -h, --help            show this help message and exit
  -v                    logging level set to ERROR
  -vv                   logging level set to INFO
  -vvv                  logging level set to DEBUG
  -g NGAMES, --games NGAMES
                        number of games for the agent to play. (default: 100)
                        NOTE: if you enable training, this variable will not
                        be used
  -m NMATCHES, --matches NMATCHES
                        number of matches for each game. (default: 5) NOTE: if
                        you enable training, this variable will not be used
  -t, --train           allows the training of the deep neural network. NOTE:
                        leave disabled to only see the current agent behave
  -c, --clear           clears the folders where the state of the agent is
                        saves. NOTE: use this to retrain the agent from
                        scratch
```

The commands are somewhat self explanatory, but let me guide you through some common things you might want to do with this application.

### Play against the CPU player?

To give it a try in the game of pong just run the following command:

```bash
python king_pong.py
```

This will fire up the CPU player on the left and you on the right. Just use the normal UP/DOWN arrow keys.

### Let them play against each other

To use the trained agent against the CPU player you can use this command:

```bash
python agent.py -g 3 -m 2
```

This will set the CPU on the left and the agent on the right. They will play 3 games of 2 points each. If you want to see log printed to the screen just add `-vv` option which gives a good amount of info, including scores.

### Continue training the agent

If you want to continue improving the agent run:

```bash
python agent.py -t -vv
```

This will continue adjusting the weights of the neural network and saving progress every so often. The networks will be saved on `networks/` directory and the references of the network actually being utilized could be found on the `networks/checkpoint` file.

### To train the agent from scratch

If you'd like to try out and train the agent yourself, do:

```bash
python agent.py -t -c -vv
```

This will first remove all the previous networks and start the training. Beware, it takes a long time to see some good progress. Couple of days to a week with a decent GPU.

## Future features

### Play against the Agent?
It would be great to be able to play against the agent. Need to:
   - Refactor `king_pong.py` to not use cpu vs agent but a generic left vs right
   - Create interface that Agent would base on
   - Abstract actions of a human and cpu player

### Agent can have dreams?
One idea for training the Deep Network is to allow the agent to train by prioritized recollection of frames. For example:
   - From the batch of frames, assign probabilities depending on absolute value of rewards.
   - From this new distribution, select frames probabilisticaly with replacement.
   - Recollect aproximatelly 100 frames per dream. About 50 before the recollected frame, about 50 after.
   - Replay longer batches but fewer times.

### Train the same agent for other games?
The Agent is a very generic class that basically takes inputs and select outputs based on those inputs. One great add to this codebase would be to make the agent so generic such that it could learn to play other games.

### Shouldn't we add some tests?
Just as a good practice some functions would be best if tested.

## More Information

This project was created to fulfill the Capstone project of the Udacity Nanodegree program. To read the report and have more details read [the report.](report.md)

## References

The following resources were particularly useful for the completion of this project:
- [Deep Learning Flappy](https://github.com/yenchenlin/DeepLearningFlappyBird)
- [Deep Learning Tutorial](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
- [Pong PyGame](https://www.youtube.com/watch?v=x_tPvtyB1fY)
- [Udacity's Reinforcement Learning Course](https://www.udacity.com/course/reinforcement-learning--ud600)

[king]: ./imgs/king.gif "King Pong Game"
[rl]: ./imgs/rl.png "Reinforcement Learning"
[dl]: ./imgs/dl.png "Deep Learning"
[drl]: ./imgs/drl.png "Deep Reinforcement Learning"

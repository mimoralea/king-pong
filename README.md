# King Pong
### Deep Reinforcement Learning Pong Player

## Overview

## What is Deep Reinforcement Learning?

### What is Deep Learning?

### What is Reinforcement Learning?


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

```
sudo dnf install opencv-python
```

### Docker image

To make things a little easier, I have created a Docker image that will hopefully help with running the code. You can either `pull` the image I have pushed into Docker.io, or you can build it from scratch.

#### To pull execute:

```
docker pull mimoralea/king-pong:v1
```

This command will download the container with all of the dependencies installed.


#### To build execute:

```
docker build -t mimoralea/king-pong:v1 .
```

This command can take a little while, but it will build a Fedora image and install all the depencies for you.

#### To get into the container:

```
docker run --privileged=true --rm \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/king-pong \
    -it mimoralea/king:v1 /bin/bash
```

This command redirects the X11 session to your local computer as well as it maps the local directory with the source code to the `/king` directory in the container.

### Run the app:

```
cd king-pong/
python agent.py -vv
```

The Pygame window should get redirected to you screen. This worked seamlessly on my Fedora environment, let me know if there is are different steps on other OS.

## Play against the CPU player?

## Play against the Agent?

## Let them play against each other

## More Information

## References

The following resources were particularly useful for the completion of this project:
- [Deep Learning Flappy](https://github.com/yenchenlin/DeepLearningFlappyBird)
- [Deep Learning Tutorial](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
- [Pong PyGame](https://www.youtube.com/watch?v=x_tPvtyB1fY)
- [Udacity's Reinforcement Learning Course](https://www.udacity.com/course/reinforcement-learning--ud600)

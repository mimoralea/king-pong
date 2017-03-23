import tensorflow as tf
import cv2
import numpy as np

class MultilayerConvolutionalNetwork:
    """
    This class manages the deep neural network
    that will be used by the agent to learn
    and extrapolate the state space
    """
    def __init__(self, input_width, input_height, nimages, nchannels):
        self.session = tf.InteractiveSession()
        self.input_width = input_width
        self.input_height = input_height
        self.nimages = nimages
        self.nchannels = nchannels
        self.a = tf.placeholder("float", [None, self.nchannels])
        self.y = tf.placeholder("float", [None])
        self.input_image, self.y_conv, self.h_fc1, self.train_step = self.build_network()
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def weight_variable(self, shape, stddev = 0.01):
        """
        Initialize weight with slight amount of noise to
        break symmetry and prevent zero gradients
        """
        initial = tf.truncated_normal(shape, stddev = stddev)
        return tf.Variable(initial)

    def bias_variable(self, shape, value = 0.01):
        """
        Initialize ReLU neurons with slight positive initial
        bias to avoid dead neurons
        """
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride = 1):
        """
        We use a stride size of 1 and zero padded convolutions
        to ensure we get the same output size as it was our input
        """
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        """
        Our pooling is plain old max pooling over 2x2 blocks
        """
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                              strides = [1, 2, 2, 1], padding = "SAME")

    def build_weights_biases(self, weights_shape):
        """
        Build the weights and bias of a convolutional layer
        """
        return self.weight_variable(weights_shape), \
            self.bias_variable(weights_shape[-1:])

    def convolve_relu_pool(self, nn_input, weights_shape, stride = 4, pool = True):
        """
        Convolve the input to the network with the weight tensor,
        add the bias, apply the ReLU function and finally max pool
        """
        W_conv, b_conv = self.build_weights_biases(weights_shape)
        h_conv = tf.nn.relu(self.conv2d(nn_input, W_conv, stride) + b_conv)
        if not pool:
            return h_conv
        return self.max_pool_2x2(h_conv)

    def build_network(self):
        """
        Sets up the deep neural network
        """

        # the input is going to be reshaped to a
        # 80x80 color image (4 channels)
        input_image = tf.placeholder("float", [None, self.input_width,
                                      self.input_height, self.nimages])

        # create the first convolutional layers
        h_pool1 = self.convolve_relu_pool(input_image, [8, 8, self.nimages, 32])
        h_conv2 = self.convolve_relu_pool(h_pool1, [4, 4, 32, 64], 2, False)
        h_conv3 = self.convolve_relu_pool(h_conv2, [3, 3, 64, 64], 1, False)

        # create the densely connected layers
        W_fc1, b_fc1 = self.build_weights_biases([5 * 5 * 64, 512])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 5 * 5 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # finally add the readout layer
        W_fc2, b_fc2 = self.build_weights_biases([512, self.nchannels])
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        readout_action = tf.reduce_sum(tf.mul(readout, self.a), reduction_indices=1)
        cost_function = tf.reduce_mean(tf.square(self.y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-8).minimize(cost_function)

        return input_image, readout, h_fc1, train_step

    def train(self, value_batch, action_batch, state_batch):
        """
        Does the actual training step
        """
        self.train_step.run(feed_dict = {
            self.y : value_batch,
            self.a : action_batch,
            self.input_image : state_batch
        })

    def save_variables(self, a_file, h_file, stack):
        """
        Saves neural network weight variables for
        debugging purposes
        """
        readout_t = self.readout_act(stack)
        a_file.write(",".join([str(x) for x in readout_t]) + '\n')
        h_file.write(",".join([str(x) for x in self.h_fc1.eval(
            feed_dict={self.input_image:[stack]})[0]]) + '\n')

    def save_percepts(self, path, x_t1):
        """
        Saves an image array to visualize
        how the image is compressed before saving
        """
        cv2.imwrite(path, np.rot90(x_t1))

    def save_network(self, directory, iteration):
        """
        Saves the progress of the agent
        for further use later on
        """
        self.saver.save(self.session, directory + '/network', global_step = iteration)

    def attempt_restore(self, directory):
        """
        Restors the latest file saved if
        available
        """
        checkpoint = tf.train.get_checkpoint_state(directory)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            return checkpoint.model_checkpoint_path

    def preprocess_percepts(self, x_t1_colored, reshape = True):
        """
        The raw image arrays get shrunk down and
        remove any color whatsoever. Also gets it in
        3 dimensions if needed
        """
        x_t1_resized = cv2.resize(x_t1_colored, (self.input_width, self.input_height))
        x_t1_greyscale = cv2.cvtColor(x_t1_resized, cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1_greyscale, 1, 255, cv2.THRESH_BINARY)
        """
        import time
        timestamp = int(time.time())
        cv2.imwrite("percepts/%d-color.png" % timestamp,
                    np.rot90(x_t1_colored))
        cv2.imwrite("percepts/%d-resized.png" % timestamp,
                    np.rot90(x_t1_resized))
        cv2.imwrite("percepts/%d-greyscale.png" % timestamp,
                    np.rot90(x_t1_greyscale))
        cv2.imwrite("percepts/%d-bandw.png" % timestamp,
                    np.rot90(x_t1))
        """
        if not reshape:
            return x_t1
        return np.reshape(x_t1, (80, 80, 1))

    def readout_act(self, stack):
        """
        Gets the best action
        for a given stack of images
        """
        stack = [stack] if hasattr(stack, 'shape') and len(stack.shape) == 3 else stack
        return self.y_conv.eval(feed_dict = {self.input_image: stack})

    def select_best_action(self, stack):
        """
        Selects the action with the
        highest value
        """
        return np.argmax(self.readout_act(stack))


def main():
    print('This module should be imported')
    pass

if __name__ == "__main__":
    main()

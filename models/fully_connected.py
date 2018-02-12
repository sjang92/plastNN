import tensorflow as tf
from .base_neural_network import BaseNeuralNetwork


class FullyConnected(BaseNeuralNetwork):

    def __init__(self, session, graph, config, dim, lr):
        self.dim = dim
        self.lr = lr
        super(FullyConnected, self).__init__(session, graph, config)

    def _define_inputs(self):
        # input dim is [1, dim] since we don't batch process
        self.input = tf.placeholder(dtype=tf.float32, shape=[1, self.dim])
        self.label = tf.placeholder(dtype=tf.int32, shape=[1])  # +1 or 0

    def _define_graph(self):
        """
        for now, don't use GPU
        """
        with self.graph.as_default():

            self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.05)
            # Note that all layers use xavier init by default

            # First layer projects onto 64.
            self.layer1 = tf.layers.dense(self.input, 64, activation=tf.nn.relu, kernel_regularizer=self.regularizer)

            # Second layer projects onto 128.
            self.layer2 = tf.layers.dense(self.layer1, 64, activation=tf.nn.relu, kernel_regularizer=self.regularizer)

            # Third layer projects onto 32
            self.layer3 = tf.layers.dense(self.layer2, 16, activation=tf.nn.relu, kernel_regularizer=self.regularizer)

            #self.layer4 = tf.layers.dense(self.layer3, 16, activation=tf.nn.relu, kernel_regularizer=self.regularizer)

            # Last layer is a logits layer
            self.logits = tf.nn.softmax(tf.layers.dense(self.layer3, 2))

            self.output = tf.argmax(self.logits, axis=1)
            self.one_hot = tf.one_hot(indices=self.label, depth=2)

    def _define_loss(self):
        with self.graph.as_default():
            self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.one_hot
            )

            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
            self.loss += reg_term

        return self.loss

    def _define_learning_rate(self):
        return self.lr

    def _define_optimizer(self):
        with self.graph.as_default():
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            self.optimizer_op = self.optimizer.minimize(self.loss)

        return self.optimizer_op

    def _define_init_op(self):
        return tf.global_variables_initializer()

    def train(self, input, label):
        # Here we have to encapsulate in list since tf takes in batches
        fd = {
            self.input: input,
            self.label: label
        }
        logit, label, loss, _ = self.session.run([self.logits, self.one_hot, self.loss, self.optimizer_op], feed_dict=fd)
        return logit, label, loss

    def evaluate(self, input, label):
        fd = {
            self.input: input,
        }
        output = self.session.run(self.output, feed_dict=fd)[0]
        label = label[0]  # label is wrapped

        if label == 1 and output == 1:
            return 0
        elif label == 1 and output == 0:
            return 2
        elif label == 0 and output == 1:
            return 1
        elif label == 0 and output == 0:
            return 3

        assert False, "Shouldn't get here"

    def assign_label(self, input):
        fd = {self.input : input,}
        output = self.session.run([self.output, self.logits], feed_dict=fd)
        return output

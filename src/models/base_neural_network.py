from collections import OrderedDict
import tensorflow as tf
from abc import abstractmethod

class BaseNeuralNetwork(object):

    def __init__(self, session, graph):

        # Tensorflow related instance variables for traversal, logging, etc
        self.session = session
        self.graph = graph
        self.train_metrics = OrderedDict()

        # Explicitly define required fields for visibility
        self.init_op = None  # param initializer must be explicitly defined by the child class
        self.variables = []

        # Define the neural network model
        self._define_model()

        self.saver = tf.train.Saver(max_to_keep=None)

    # 1. Model building related interface (Private)

    @abstractmethod
    def _define_inputs(self): pass

    @abstractmethod
    def _define_graph(self): pass

    @abstractmethod
    def _define_loss(self): pass

    @abstractmethod
    def _define_optimizer(self): pass

    @abstractmethod
    def _define_learning_rate(self): pass

    def _define_init_op(self):
        return tf.global_variables_initializer()

    def _define_model(self):
        """
        Basic interface for a simple neural network model.
        For any type of neural network, we ask our client to define
            1) input placeholders
            2) graph : layers of the network
            3) loss : loss function to optimize
            4) optimizer : what kind of optimizer to use, how to optimize, etc
            5) initializer : how to initialize the network parameters
        """

        # define global step and learning rate
        self.step = tf.Variable(0, dtype=tf.int64, name='step')
        self.lr = self._define_learning_rate()

        self._define_inputs()
        self._define_graph()
        self.loss = self._define_loss()
        self.optimizer_op = self._define_optimizer()
        self.init_op = self._define_init_op()

    # 2. Output / Train process related interface (Public)

    @abstractmethod
    def train_batch(self, inputs, labels): pass

    # 3. Parameter Saving / Restoring / Initializing interface (Public)

    def save(self, model_path):
        assert self.saver is not None, "self.saver must be defined"
        assert self.session is not None, "self.session must be defined"

        self.saver.save(self.session, model_path)

    def load(self, model_path):
        assert self.saver is not None, "self.saver must be defined"
        assert self.session is not None, "self.session must be defined"

        # Restore parameters from the model path
        self.saver.restore(self.session, model_path)

    def init(self):
        """
        run the pre-defined initializer operation to initialize all parameters
        """
        self.session.run(self.init_op)

    # 4. Logging / Metrics visibility interface

    def _define_metrics(self):
        assert self.loss is not None, "self.loss must be defined"
        assert self.step is not None, "self.step must be defined"
        assert self.lr is not None, "self.lr must be defined"

        # By default loss is the only metric that we care about
        self.train_metrics['step'] = self.step
        self.train_metrics['loss'] = self.loss
        self.train_metrics['lr'] = self.lr

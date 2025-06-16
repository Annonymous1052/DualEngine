"""CLI wrapper for tflite_transfer_converter.

Converts a TF model to a TFLite transfer learning model.
"""

import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam

NUM_STATES = 15
NUM_ACTIONS = 81
BATCH_SIZE = 16
DISCOUNT_FACTOR = 0.95
lr = 0.01
tf.compat.v1.enable_v2_behavior()


def build_model():
    # H&M
    builtModel = keras.Sequential()
    builtModel.add(layers.Dense(NUM_ACTIONS, input_dim=NUM_STATES, activation='linear', kernel_initializer='normal',
                                trainable=True))
    builtModel.summary()
    builtModel.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return builtModel


class TransferLearningModel(tf.Module):
    """TF Transfer Learning model class."""

    def __init__(self, learning_rate=0.001):
        self.num_features = NUM_STATES
        self.num_classes = NUM_ACTIONS
        self.batch_size = BATCH_SIZE
        self.discount_factor = DISCOUNT_FACTOR
        self.model = build_model()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.target_model = build_model()

    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_STATES], tf.float32),
        tf.TensorSpec([None, NUM_ACTIONS], tf.float32),
        tf.TensorSpec([None, NUM_ACTIONS], tf.float32),
        tf.TensorSpec([None, NUM_STATES], tf.float32),
    ])
    def train(self, state, action, reward, next_state):
        """Runs one training step with the given samples.
    """
        with tf.GradientTape() as tape:
            prediction = self.model(state)
            prediction_next = self.target_model(next_state)
            reversed_action = tf.where(tf.equal(action, 1), tf.zeros_like(action), tf.ones_like(action))
            prediction_0 = tf.math.multiply(reversed_action, prediction)  # prediction with selected action=0 [3.5, 0, 4, 5]
            prediction_next_0 = tf.math.multiply(action, prediction_next)  # prediction with selected action=x [0. x, 0, 0]
            discount_next = tf.math.multiply(self.discount_factor, prediction_next_0)
            observation = tf.add(tf.add(prediction_0, reward), discount_next)
            loss = self.loss_fn(observation, prediction)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_STATES], tf.float32)
    ])
    def infer(self, state):
        output = self.model(state)
        return {'output': output}

    @tf.function(input_signature=[
        tf.TensorSpec([None, NUM_STATES], tf.float32)
    ])
    def infer_target(self, state):
        output = self.target_model(state)
        return {'output': output}

    # @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def update_target(self, tmp):
        self.target_model.weights[0].assign(self.model.weights[0])
        self.target_model.weights[1].assign(self.model.weights[1])
        return tf.constant(1.0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.float32)])
    def update_lr(self, tmp):
        self.optimizer.learning_rate = tmp
        self.optimizer.lr = tmp
        return tf.constant(1.0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def return_weights(self, checkpoint_path):
        # checkpoint_path = np.array("/data/local/tmp/model_saved.ckpt")
        # tensor_names = [weight.name for weight in self.model.weights]
        # tensors_to_save = [weight.read_value() for weight in self.model.weights]
        # tf.raw_ops.Save(
        #     filename=checkpoint_path, tensor_names=tensor_names,
        #     data=tensors_to_save, name='save')
        # weights = []
        # for layer in self.model.layers:
        #     weights.append(layer.get_weights())
        # print(self.model.weights[0])
        return self.model.weights
        # return tf.constant(1.0)

    @tf.function(input_signature=[
        tf.TensorSpec([NUM_STATES, NUM_ACTIONS], tf.float32),
        tf.TensorSpec([NUM_ACTIONS], tf.float32),
    ])
    def restore(self, weight, bias):
        self.model.weights[0].assign(weight)
        self.model.weights[1].assign(bias)
        return tf.constant(1.0)


def convert_and_save(saved_model_dir='saved_model'):
    """Converts and saves the TFLite Transfer Learning model.

  Args:
    saved_model_dir: A directory path to save a converted model.
  """
    model = TransferLearningModel()

    print("============================")
    print("testing infer..")
    state1 = np.array([[1., 1., 1., 1., 1., 100., 1., 1., 1., 1., 1., 1., 1., 15., 1.]])
    prediction1 = model.infer(state1)
    print(len(prediction1['output'][0]))
    print(np.argmax(prediction1['output']))

    print("============================")
    print("testing train..")
    action1 = np.zeros([1, NUM_ACTIONS], float)
    action1[0] = 1.0

    reward1 = np.ones([1, NUM_ACTIONS], float)
    reward1[0] = 3.0

    next_state1 = np.ones([1, NUM_STATES], float)
    next_state1[0] = 2.0

    loss1 = model.train(state1, action1, reward1, next_state1)
    print("loss", loss1)

    print("============================")
    print("testing update lr..")
    print(model.update_lr(0.1))
    # print("============================")
    # print("testing restore..")
    # result1 = model.infer(state1)
    # saved_weights = model.return_weights(checkpoint_path=np.array("tmp", dtype=np.string_))
    # print(saved_weights[0])
    # print(saved_weights[1])
    # model.restore(saved_weights[0], saved_weights[1])
    # result2 = model.infer(state1)
    # saved_weights2 = model.return_weights(checkpoint_path=np.array("tmp", dtype=np.string_))
    # print(result1, result2)
    # print(np.array_equal(result1, result2))

    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={
            'infer': model.infer.get_concrete_function(),
            'train': model.train.get_concrete_function(),
            'return_weights': model.return_weights.get_concrete_function(),
            'restore': model.restore.get_concrete_function(),
            'update_target': model.update_target.get_concrete_function(),
            'infer_target': model.infer_target.get_concrete_function(),
            'update_lr': model.update_lr.get_concrete_function(),
        })
    #

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # test
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    # print(interpreter.get_signature_list())
    print("printing keys..")
    signature_list = interpreter.get_signature_list()
    for i in signature_list:
        print("function key:", i)
        print(signature_list[i])
        print()


    # saver = interpreter.get_signature_runner('return_weights')
    # result = saver(checkpoint_path=np.array("tmp", dtype=np.string_))
    # print(result)

    # save model
    model_file_path = os.path.join('model.tflite')
    with open(model_file_path, 'wb') as model_file:
        model_file.write(tflite_model)


if __name__ == '__main__':
    convert_and_save()

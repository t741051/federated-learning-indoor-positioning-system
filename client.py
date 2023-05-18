import argparse
import os
from pathlib import Path
import numpy as np

import tensorflow as tf

import flwr as fl

from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    # def __init__(self, model, x_train, y_train, x_test, y_test):
    #     self.model = model
    #     self.x_train, self.y_train = x_train, y_train
    #     self.x_test, self.y_test = x_test, y_test

    def __init__(self, model, train_generator, test_generator):
        self.model = model
        self.train_generator = train_generator
        self.test_generator = test_generator

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception(
            "Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        # history = self.model.fit(
        #     self.x_train,
        #     self.y_train,
        #     batch_size,
        #     epochs,
        #     validation_split=0.1,
        # )
        history = self.model.fit(
            self.train_generator,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=self.test_generator
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        # num_examples_train = len(self.x_train)
        num_examples_train = len(self.train_generator[0])

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        # loss, accuracy = self.model.evaluate(
        #     self.x_test, self.y_test, 32, steps=steps)
        # loss, accuracy = self.model.evaluate_generator(
        #     self.test_generator, steps=len(self.test_generator))
        loss, accuracy = self.model.evaluate(
            self.test_generator, steps=len(self.test_generator))

        # num_examples_test = len(self.x_test)
        num_examples_test = len(self.test_generator)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of CIFAR10 to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    # Load and compile Keras model
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )

    # model = tf.keras.applications.MobileNet(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )

    base_model = MobileNet(input_shape=(
        75, 75, 3), include_top=False, weights='imagenet')

    # 添加自定義的輸出層
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(17, activation='softmax')(x)

    # 構建完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 鎖定 InceptionResNetV2 的卷積層權重，只訓練自定義的輸出層
    for layer in base_model.layers:
        layer.trainable = False

    # model.compile("adam", "sparse_categorical_crossentropy",
    #               metrics=["accuracy"])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Load a subset of CIFAR-10 to simulate the local data partition
    # (x_train, y_train), (x_test, y_test) = load_partition(args.partition)
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # 对图像进行缩放，将像素值归一化到[0,1]之间
        rotation_range=30,  # 随机旋转角度范围
        width_shift_range=0.2,  # 随机水平平移的范围
        height_shift_range=0.2,  # 随机竖直平移的范围
        shear_range=0.1,  # 剪切强度
        zoom_range=0.2,  # 随机缩放范围
        horizontal_flip=False,  # 随机水平翻转
        fill_mode='nearest'  # 填充方式
    )

    train_dir = 'Dataset/train/'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'Dataset/test',
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    x_train = np.load('train_image.npy')
    y_train = np.load('train_label.npy')

    x_test = np.load('test_image.npy')
    y_test = np.load('test_label.npy')

    if args.toy:
        x_train, y_train = x_train[:10], y_train[:10]
        x_test, y_test = x_test[:10], y_test[:10]

    # Start Flower client
    # client = CifarClient(model, x_train, y_train, x_test, y_test)
    client = CifarClient(model, train_generator, test_generator)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000: (idx + 1) * 5000],
        y_train[idx * 5000: (idx + 1) * 5000],
    ), (
        x_test[idx * 1000: (idx + 1) * 1000],
        y_test[idx * 1000: (idx + 1) * 1000],
    )


if __name__ == "__main__":
    main()

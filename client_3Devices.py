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

    def __init__(self, model, train_generator):
        self.model = model
        self.train_generator = train_generator

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

        history = self.model.fit(
            self.train_generator,
            batch_size=batch_size,
            epochs=epochs,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.train_generator[0])

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            # "val_loss": history.history["val_loss"][0],
            # "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    # def evaluate(self, parameters, config):
    #     """Evaluate parameters on the locally held test set."""

    #     # Update local model with global parameters
    #     self.model.set_weights(parameters)

    #     # Get config values
    #     # steps: int = config["val_steps"]

    #     # Evaluate global model parameters on the local test data and return results
    #     # loss, accuracy = self.model.evaluate(
    #     #     self.x_test, self.y_test, 32, steps=steps)

    #     loss, accuracy = self.model.evaluate(
    #         self.val_generator, steps=len(self.val_generator))

    #     # num_examples_test = len(self.x_test)
    #     num_examples_test = len(self.val_generator)
    #     return loss, num_examples_test, {"accuracy": accuracy}


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

    args = parser.parse_args()

    # Load and compile Keras model

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
    if (args.partition == 0):
        train_dir = 'Dataset_3Devices/Nokia/train/'
    elif (args.partition == 1):
        train_dir = 'Dataset_3Devices/HTC/train/'
    elif (args.partition == 2):
        train_dir = 'Dataset_3Devices/Mi/train/'

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical'
    )

    # Start Flower client

    client = CifarClient(model, train_generator)

    ip_address = '192.168.0.21'  # here you should write the server ip-address
    server_address=ip_address + ':8080'

    fl.client.start_numpy_client(
        # server_address="127.0.0.1:8080",
        server_address=server_address,
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
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

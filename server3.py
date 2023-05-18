from typing import Dict, Optional, Tuple
from pathlib import Path

import flwr as fl
import numpy as np
import tensorflow as tf

from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    # model = tf.keras.applications.MobileNet(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    # model = tf.keras.applications.MobileNet(
    #     input_shape=(75, 75, 3), weights=None, classes=17
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

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # 创建验证集的 ImageDataGenerator 对象
    validation_datagen = ImageDataGenerator(rescale=1./255)

    val_dir = 'Dataset_3Devices/val/'

    # 加载验证集
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(75, 75),
        batch_size=32,
        class_mode='categorical'
    )
    x_val = np.load('val_image.npy')
    y_val = np.load('val_label.npy')
    # The `evaluate` function will be called after every round
    print("y_val", y_val.shape)

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        model.set_weights(parameters)
        # loss, accuracy = model.evaluate(x_val, y_val)
        # loss, accuracy = model.evaluate_generator(validation_generator)
        loss, accuracy = model.evaluate(validation_generator)

        best_accuracy = -np.inf  # 賦初值

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save("model.h5")
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()

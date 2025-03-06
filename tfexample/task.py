"""tfexample: A Flower / TensorFlow app."""

import os

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# def load_model(learning_rate: float = 0.001):
#     # Define a simple CNN for CIFAR-10 and set Adam optimizer
#     model = keras.Sequential(
#         [
#             keras.Input(shape=(32, 32, 3)),
#             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#             layers.MaxPooling2D(pool_size=(2, 2)),
#             layers.Flatten(),
#             layers.Dropout(0.5),
#             layers.Dense(10, activation="softmax"),
#         ]
#     )
#     optimizer = keras.optimizers.Adam(learning_rate)
#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model
from keras import layers, models, optimizers

def load_model(learning_rate: float = 0.001):
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    encoded = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder model
    autoencoder = models.Model(input_layer, decoded)
    optimizer = optimizers.Adam(learning_rate)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    return autoencoder


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on ea.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    return x_train, y_train, x_test, y_test


# # Load trained model
# autoencoder = load_model()
# autoencoder.load_weights("trained_autoencoder.h5")  # Load trained weights

# # Predict reconstruction error
# reconstructed = autoencoder.predict(x_test)
# mse_errors = np.mean(np.square(x_test - reconstructed), axis=(1, 2, 3))

# # Set a threshold (e.g., 95th percentile of MSE)
# threshold = np.percentile(mse_errors, 95)

# # Flag anomalies
# anomalies = mse_errors > threshold
# print(f"Detected {sum(anomalies)} anomalies")

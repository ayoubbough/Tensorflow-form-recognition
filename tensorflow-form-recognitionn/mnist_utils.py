import numpy as np
import tensorflow as tf
from tensorflow import keras


class DataSet:
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self._images = images
        self._labels = labels
        self._epochs_completed: int = 0
        self._index_in_epoch: int = 0
        self._num_examples: int = images.shape[0]

    @property
    def images(self) -> np.ndarray:
        return self._images

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        start: int = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Fin de l'époque
            self._epochs_completed += 1
            # Mélange des données
            perm: np.ndarray = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Début de la prochaine époque
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end: int = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSets:
    def __init__(self, train: DataSet, test: DataSet) -> None:
        self.train: DataSet = train
        self.test: DataSet = test


def read_data_sets() -> DataSets:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation et redimensionnement
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # Encodage One-hot
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return DataSets(DataSet(x_train, y_train), DataSet(x_test, y_test))


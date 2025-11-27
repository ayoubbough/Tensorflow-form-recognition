
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import Constant, TruncatedNormal
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
)

from mnist_utils import read_data_sets


def build_model() -> keras.Model:
    inputs = Input(shape=(784,), name="input_tensor")
    
    x = Reshape((28, 28, 1))(inputs)
    
    x = Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        use_bias=True,
        bias_initializer=Constant(0.1),
        name="conv2d_1",
    )(x)
    x = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool_1"
    )(x)
    x = Conv2D(
        64,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        use_bias=True,
        bias_initializer=Constant(0.1),
        name="conv2d_2",
    )(x)
    x = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool_2"
    )(x)
    x = Flatten(name="flatten")(x)
    x = Dense(
        1024,
        activation="relu",
        kernel_initializer=TruncatedNormal(stddev=0.1),
        use_bias=True,
        bias_initializer=Constant(0.1),
        name="fc1",
    )(x)
    x = Dropout(0.5, name="dropout")(x)
    outputs = Dense(
        10,
        kernel_initializer=TruncatedNormal(stddev=0.1),
        use_bias=True,
        bias_initializer=Constant(0.1),
        name="output_tensor",
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(
    output_dir: str = "./model_keras", steps: int = 1000
) -> tuple[keras.Model, tf.Tensor]:
    mnist_data = read_data_sets()
    x_train = mnist_data.train.images
    y_train = mnist_data.train.labels
    x_test = mnist_data.test.images
    y_test = mnist_data.test.labels

    model = build_model()
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Entraînement
    batch_size = 50
    
    # Création d'un tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(60000).batch(batch_size).repeat()
    
    model.fit(train_ds, epochs=1, steps_per_epoch=steps, verbose=1)
    
    # Sauvegarde
    try:
        model.export(output_dir)
    except AttributeError:
        # Si la version de Keras ne supporte pas l'export, méthode alternative
        tf.saved_model.save(model, output_dir)
    except Exception as e:
         print(f"Could not save model: {e}")

    # Évaluation
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"test accuracy {test_acc}")
    
    return model, model.output


if __name__ == "__main__":
    train_model()

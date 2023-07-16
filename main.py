import tensorflow as tf


def tutorial():
    """Tutorial from https://www.tensorflow.org/tutorials/quickstart/beginner"""

    # Load a dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build a machine learning model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # predictions = model(x_train[:1]).numpy()
    # print(predictions)

    # tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # print(loss_fn(y_train[:1], predictions).numpy())

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Train and evaluate your model

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


def lenet_5():
    """Usage of tensorflow toolset to recreate the model of LeNet-5, modified to utilize ReLU activation."""

    # Load a dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build a machine learning model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6,
                               kernel_size=(5, 5),
                               padding='valid',
                               input_shape=(28, 28, 1),
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(5, 5),
                               padding='valid',
                               input_shape=(14, 14, 6),
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                  strides=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Train and evaluate your model
    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


if __name__ == '__main__':
    print('TensorFlow version: ', tf.__version__)
    # tutorial()
    lenet_5()

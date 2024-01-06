import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split

def load_cifar100_data(batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size, random_state=random_seed)

    if test:
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return test_dataset

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=50000, seed=random_seed)

    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, valid_dataset

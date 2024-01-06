import tensorflow as tf
from tensorflow.keras import layers

class VGG19(tf.keras.Model):
    def __init__(self, num_classes=100):
        super(VGG19, self).__init__()
        self.conv_layers = [
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        ]
        self.flatten = layers.Flatten()
        self.fc_layers = [
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dense(num_classes)
        ]

    def call(self, x, training=False):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.fc_layers:
            x = layer(x)
        return x

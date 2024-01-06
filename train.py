import tensorflow as tf
from data_loader import load_cifar100_data
from vgg19_model import VGG19

class CustomCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.1, name='custom_cross_entropy_loss'):
        super(CustomCrossEntropyLoss, self).__init__(name=name)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]

        # Apply label smoothing
        y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        # Calculate cross-entropy loss
        loss = -tf.reduce_sum(y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)))

        return loss

# Device configuration
device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'

# CIFAR100 dataset
train_loader, valid_loader = load_cifar100_data(batch_size=64)

test_loader = load_cifar100_data(batch_size=64, test=True)

num_classes = 100
num_epochs = 20
batch_size = 16
learning_rate = 0.005

model = VGG19(num_classes)

# Loss and optimizer
criterion = CustomCrossEntropyLoss(label_smoothing=0.1)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        with tf.device(device):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                loss = criterion(labels, outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.numpy()))

    correct = 0
    total = 0
    for images, labels in valid_loader:
        with tf.device(device):
            outputs = model(images, training=False)
            predicted = tf.argmax(outputs, axis=1)
            total += labels.shape[0]
            correct += tf.reduce_sum(tf.cast(predicted == labels, tf.float32)).numpy()

    print('Accuracy on the validation images: {:.2f} %'.format(100 * correct / total))

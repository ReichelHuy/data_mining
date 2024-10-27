import numpy as np
import tensorflow as tf

seed = 42

def load_mnist():
    # Tải dữ liệu MNIST
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Chia dữ liệu thành tập huấn luyện và tập xác thực
    # 10% dữ liệu huấn luyện sẽ được sử dụng cho xác thực
    validation_size = int(0.1 * train_images.shape[0])
    val_images = train_images[:validation_size]
    val_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # Tiền xử lý dữ liệu
    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    train_images = train_images.reshape(train_images.shape[0], -1)  # (num_samples, 784)
    val_images = val_images.reshape(val_images.shape[0], -1)        # (num_samples, 784)
    test_images = test_images.reshape(test_images.shape[0], -1)     # (num_samples, 784)

    train_labels = vectorized_result(train_labels)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def vectorized_result(y):
    b = np.zeros((y.size, y.max() + 1))
    b[np.arange(y.size), y] = 1
    return b

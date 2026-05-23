import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

E = 2.718281828459045


def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class ConvLayer:

    def __init__(self, kernel_size):
        self.kernel = np.random.randn(kernel_size, kernel_size) * 0.01

    def forward(self, image):
        self.image = image
        h, w = image.shape
        k = self.kernel.shape[0]
        output = np.zeros((h - k + 1, w - k + 1))

        for i in range(h - k + 1):
            for j in range(w - k + 1):
                region = image[i : i + k, j : j + k]
                output[i, j] = np.sum(region * self.kernel)
        return output


class MaxPool:

    def forward(self, image):
        h, w = image.shape
        out_h = h // 2
        out_w = w // 2
        output = np.zeros((out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                region = image[i * 2 : i * 2 + 2, j * 2 : j * 2 + 2]
                output[i, j] = np.max(region)
        return output


class FullyConnected:

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias


class CNN:

    def __init__(self):
        self.conv1 = ConvLayer(3)
        self.pool1 = MaxPool()
        self.fc1 = FullyConnected(13 * 13, 10)

    def forward(self, image):
        x = self.conv1.forward(image)
        x = relu(x)
        x = self.pool1.forward(x)
        self.features = x.flatten()
        x = self.fc1.forward(self.features)
        x = softmax(x)
        return x
    
    def train(self, image, label, lr):
        output = self.forward(image)
        target = np.zeros(10)
        target[label] = 1
        loss = -np.log(output[label] + 1e-15)
        error = output - target
        dW = np.outer(self.features, error)
        db = error
        self.fc1.weights -= lr * dW
        self.fc1.bias -= lr * db
        return loss

def load_mnist_dataset(path, limit=100):
    images_path = os.path.join(path, "train-images-idx3-ubyte")
    labels_path = os.path.join(path, "train-labels-idx1-ubyte")

    with open(images_path, "rb") as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

    with open(labels_path, "rb") as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype=">i4")
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    images = images[:limit]
    labels = labels[:limit]

    images_normalized = [img / 255.0 for img in images]

    return images_normalized, labels


if __name__ == "__main__":
    cnn = CNN()
    train_images, train_labels = load_mnist_dataset("mnist_dataset",limit=1000)
    lr = 0.1
    epochs = 100
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(train_images)):
            loss = cnn.train(train_images[i],train_labels[i],lr)
            total_loss += loss
        avg_loss = total_loss / len(train_images)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss = {avg_loss:.4f}")
    correct = 0
    for i in range(len(train_images)):
        output = cnn.forward(train_images[i])
        prediction = np.argmax(output)
        if prediction == train_labels[i]:
            correct += 1
    accuracy = correct / len(train_images)
    print(f"\nAccuracy = {accuracy:.2f}")
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.show()

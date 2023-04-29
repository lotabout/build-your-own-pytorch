from network import *
from autograd import Tensor
from ops import cross_entropy
from datetime import datetime
import random

# Example of building a LeNet

# ref: https://d2l.ai/chapter_convolutional-neural-networks/lenet.html

# pip install python-mnist

def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

lenet = Sequential(
    LazyConv2d(6, kernel_size=5, padding=2), Sigmoid(),
    AvgPool2d(kernel_size=2, stride=2),
    LazyConv2d(16, kernel_size=5), Sigmoid(),
    AvgPool2d(kernel_size=2, stride=2),
    Flatten(),
    LazyLinear(120), Sigmoid(),
    LazyLinear(84), Sigmoid(),
    LazyLinear(10))


from mnist import MNIST

# download MNIST and unzip it in the data directory
# http://yann.lecun.com/exdb/mnist/
mndata = MNIST('data')
train_images, train_labels = mndata.load_training()

def gradient_descent(x, grad):
    learning_rate = 0.1
    return x - learning_rate * grad

mini_batch = 128
epochs = 3

# Note that 

for epoch in range(epochs):
    print(f'epoch: {epoch}, {datetime.now()}')
    for i in range(0, 59904, mini_batch):
        x = Tensor(np.array(train_images[i:i+mini_batch]).reshape(mini_batch, 1, 28, 28) / 255.0)
        y = Tensor(one_hot(train_labels[i:i+mini_batch], 10))
        y_pred = lenet(x)
        loss = cross_entropy(y_pred, y)
        loss.backward()
        lenet.update_gradient(gradient_descent)
        print(f'{datetime.now()} mini_batch: {i:05}: {loss.ndarray}')

def predict(image):
    print(mndata.display(image))
    x = Tensor(np.array(image).reshape(1, 1, 28, 28) / 255.0)
    y_pred_probabilities = lenet(x)
    result = np.argmax(y_pred_probabilities.ndarray, axis=1)[0]
    print(f'prediction: {result}')
    return result

print('Examples of prediction')
for i in range(10):
    choice = random.randrange(0, 10000)
    predict(test_images[choice])

print('Validating...(takes around 2 minutes)')
# validate on test set
test_images, test_labels = mndata.load_testing()
x = Tensor(np.array(test_images).reshape(10000, 1, 28, 28) / 255.0)
y_pred_probabilities = lenet(x)
predictions = np.argmax(y_pred_probabilities.ndarray, axis=1)

true_values = np.array(test_labels)
N = true_values.shape[0]
accuracy = (true_values == predictions).sum() / N
print(f'validation accuracy: {accuracy}')

print('Examples of wrong prediction')
wrong_predictions = np.where(true_values != predictions)[0]
for i in range(10):
    choice = random.randrange(0, len(wrong_predictions))
    image_index = wrong_predictions[choice]
    predict(test_images[image_index])
    print(f'Actual: {test_labels[image_index]}')

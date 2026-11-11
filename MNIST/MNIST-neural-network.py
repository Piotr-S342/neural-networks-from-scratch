import numpy as np
import gzip
import os
import matplotlib.pyplot as plt

data = {
    "training_images": "MNIST-data/train-images.idx3-ubyte",
    "training_labels": "MNIST-data/train-labels.idx1-ubyte",
    "testing_images": "MNIST-data/t10k-images.idx3-ubyte",
    "testing_labels": "MNIST-data/t10k-labels.idx1-ubyte"
}

train_images = open(data["training_images"], mode="rb")
train_labels = open(data["training_labels"], mode="rb")
test_images = open(data["testing_images"], mode="rb")
test_labels = open(data["testing_labels"], mode="rb")

mnist_dataset = {}

mnist_dataset["train_images"] = np.frombuffer(train_images.read(),np.uint8,offset = 16).reshape(-1,28 * 28)
mnist_dataset["train_labels"] = np.frombuffer(train_labels.read(),np.uint8,offset = 8)
mnist_dataset["test_images"] = np.frombuffer(test_images.read(),np.uint8,offset = 16).reshape(-1,28 * 28)
mnist_dataset["test_labels"] = np.frombuffer(test_labels.read(),np.uint8,offset = 8)

x_train = np.array(mnist_dataset["train_images"])
y_train = np.array(mnist_dataset["train_labels"])
x_train = x_train.astype('float32') / 255
y_train = np.eye(10)[y_train]

x_test = np.array(mnist_dataset["test_images"])
y_test = np.array(mnist_dataset["test_labels"])
x_test = x_test.astype('float32') / 255
y_test = np.eye(10)[y_test]

weights = np.random.randn(784,64) * np.sqrt(2 / (784 + 64) )

bias = np.zeros(64,)

out_weights = np.random.randn(64,10) * np.sqrt(2 / 74 )
out_bias = np.zeros(10,)

learning_rate = 0.1
batch_size = 64

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
def relu_dx(x):
    return (x>0).astype(float)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

def cross_entropy(prediction,target):
    return -np.mean(np.sum(target * np.log(prediction + 1e-12),axis = 1))

def forward_propagation():
    hiden_out = relu(x @ weights + bias)
    out = softmax(hiden_out @ out_weights + out_bias)

    L = cross_entropy(out, y)
    return hiden_out, out, L

def backward_propagation(hiden_out, out, L):
          #diferenciating to find how changing prediction affects the loss (if negative pred needs to increase)
    out_grad = (out - y) / x.shape[0]  #same thing as ^ but before the sigmoid ((relu now)) (uses derivative of sigmoid and the calculation before)

    weight_grad = hiden_out.T @ out_grad       #finds the effect of weight on loss using chain rule and the calculation ^
    bias_grad = np.sum(out_grad, axis = 0)               #finds effect of bias on loss

    hiden_activation_grad = out_grad @ out_weights.T
    hiden_out_grad = hiden_activation_grad * relu_dx(hiden_out)
    hiden_weight_grad = x.T @ hiden_out_grad
    hiden_bias_grad = np.sum(hiden_out_grad, axis = 0)

    return weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad

def gradient_descent(weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad,out_weights,out_bias,weights,bias):
    out_weights -= learning_rate * weight_grad
    out_bias -= learning_rate * bias_grad
    weights -= learning_rate * hiden_weight_grad
    bias -= learning_rate * hiden_bias_grad
    return out_weights, out_bias,bias,weights

for epoch in range(1000):
    total_loss = 0
    indeces = np.random.permutation(len(x_train))
    x_train = x_train[indeces]
    y_train = y_train[indeces]
    for start in range(0,len(x_train),batch_size):
        end = start + batch_size
        x = x_train[start:end]
        y = y_train[start:end]
        hiden_out, out, L = forward_propagation()
        weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad = backward_propagation(hiden_out, out, L)
        out_weights, out_bias,bias,weights = gradient_descent(weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad,out_weights,out_bias,weights,bias)
        total_loss += L * len(x)
    average_loss = total_loss / len(x_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:,}: Loss {average_loss:.6f}")


hiden_out, out, L = forward_propagation()

print(f'''
Predictions:
{out}
Final loss:
{L}
''')

np.savez(
    "MNIST_weights.npz",
    weights=weights,
    bias=bias,
    out_weights=out_weights,
    out_bias=out_bias
)
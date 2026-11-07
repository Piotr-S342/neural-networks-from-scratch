import numpy as np
import gzip
import os
import matplotlib.pyplot as plt

data = {
    "training_images": "MNIST/MNIST-data/train-images.idx3-ubyte",
    "training_labels": "MNIST/MNIST-data/train-labels.idx1-ubyte",
    "testing_images": "MNIST/MNIST-data/t10k-images.idx3-ubyte",
    "testing_labels": "MNIST/MNIST-data/t10k-labels.idx1-ubyte"
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

x = np.array(mnist_dataset["train_images"])
y = np.array(mnist_dataset["train_labels"])
x = x.astype('float32') / 255
y = np.eye(10)[y]

x_test = np.array(mnist_dataset["test_images"])
y_test = np.array(mnist_dataset["test_labels"])
x_test = x_test.astype('float32') / 255
y_test = np.eye(10)[y_test]

weights = np.random.rand(784,64) * np.sqrt(2 / 784 + 64 )

bias = np.zeros(64,)

out_weights = np.random.rand(64,10) * np.sqrt(2 / 74 )
out_bias = np.zeros(10,)

learning_rate = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation():
    hiden_out = sigmoid(x @ weights + bias)
    out = sigmoid(hiden_out @ out_weights + out_bias)

    L = np.mean((out - y) ** 2)
    return hiden_out, out, L

def backward_propagation(hiden_out, out, L):
    pred_grad = (2/(x.shape[0] * 10)) *(out - y)      #diferenciating to find how changing prediction affects the loss (if negative pred needs to increase)
    out_grad = pred_grad * (out * (1 - out))   #same thing as ^ but before the sigmoid (uses derivative of sigmoid and the calculation before)

    weight_grad = hiden_out.T @ out_grad       #finds the effect of weight on loss using chain rule and the calculation ^
    bias_grad = np.sum(out_grad, axis = 0)               #finds effect of bias on loss

    hiden_activation_grad = out_grad @ out_weights.T
    hiden_out_grad = hiden_activation_grad * (hiden_out * (1 - hiden_out))
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
    hiden_out, out, L = forward_propagation()
    weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad = backward_propagation(hiden_out, out, L)
    out_weights, out_bias,bias,weights = gradient_descent(weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad,out_weights,out_bias,weights,bias)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:,}: Loss {L:.6f}")


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
import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])


weights = np.array([
    [0.9,-0.3],
    [-0.2,0.9]
])

bias = np.array([0.,0.])

out_weights = np.array([
    [0.3],
    [0.8]
])
out_bias = np.array([0.])

# data = np.load("xor_weights.npz")
#
# weights = data["weights"]
# bias = data["bias"]
# out_weights = data["out_weights"]
# out_bias = data["out_bias"]

learning_rate = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation():
    hiden_out = sigmoid(x @ weights + bias)
    out = sigmoid(hiden_out @ out_weights + out_bias)

    L = np.mean((out - y) ** 2)
    return hiden_out, out, L

def backward_propagation(hiden_out, out, L):
    pred_grad = (2/x.shape[0]) *(out - y)      #diferenciating to find how changing prediction affects the loss (if negative pred needs to increase)
    out_grad = pred_grad * (out * (1 - out))   #same thing as ^ but before the sigmoid (uses derivative of sigmoid and the calculation before)

    weight_grad = hiden_out.T @ out_grad       #finds the effect of weight on loss using chain rule and the calculation ^
    bias_grad = np.sum(out_grad)               #finds effect of bias on loss

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

for epoch in range(1000000000):
    hiden_out, out, L = forward_propagation()
    weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad = backward_propagation(hiden_out, out, L)
    out_weights, out_bias,bias,weights = gradient_descent(weight_grad, bias_grad, hiden_weight_grad, hiden_bias_grad,out_weights,out_bias,weights,bias)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:,}: Loss {L:.6f}")

hiden_out, out, L = forward_propagation()

print(f'''
Predictions:
{out}
Final loss:
{L}
''')

np.savez(
    "xor_weights.npz",
    weights=weights,
    bias=bias,
    out_weights=out_weights,
    out_bias=out_bias
)
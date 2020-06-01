import numpy as np

feature_set = np.array([[1,0,1],[1,1,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0]])
labels = np.array([[1,1,0,0,1,0]]).T

weights = np.random.rand(3,1)
bias = np.random.rand(1)
learning_rate = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(10000):

    output = sigmoid(np.dot(feature_set, weights) + bias) 

    output_delta = (output - labels) *  sigmoid_der(output)

    weights -= learning_rate * np.dot(feature_set.T, output_delta)

    for num in output_delta:
        bias -= learning_rate * num
        
# prediction
single_point = np.array([0,1,1])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)
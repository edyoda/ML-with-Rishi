# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 19:26:55 2021

@author: RISHBANS
"""

#Initialize Learning rate, bias and weights

import math
lr = 0.1
bias = 1  #x_0
weights = [-20, 20, 20]


def sigmoid(x):
    return 1/(1+math.exp(-x))

#Function perceptron:  perceptron(x_1, x_2, output) 
#Output = x0 + x1*w1 + x2*w2 + x3w3
def perceptron(x_1, x_2, output_A):
    #fwd
    output_P = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    #output_P = sigmoid(output_P)
    if output_P > 4.6:
        output_P = 1
    else:
        output_P = 0
    #bck
    error = 1/2*(output_A - output_P)**2
    weights[0] = weights[0] + error*lr*bias
    weights[1] = weights[1] + error*lr*x_1
    weights[2] = weights[2] + error*lr*x_2

#Function predict:  predict(x_1, x_2)
def predict(x_1, x_2):
    output_P = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    if output_P > 4.6:
        output_P = 1
    else:
        output_P = 0
    return output_P


#Call perceptron for each row of OR gate
#Run in Loop for multiple times to train the Network
for i in range(0, 60):
    perceptron(0,0,0)
    perceptron(0,1,1)
    perceptron(1,0,1)
    perceptron(1,1,1)
    print(weights)



#Take Input values from user to predict the value 
print("Enter first input: x_1")
x_1 = int(input())

print("Enter second input: x_2")
x_2 = int(input())

output_predict = predict(x_1, x_2)
print(x_1, "or", x_2, "is: ", output_predict)





















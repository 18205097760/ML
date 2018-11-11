import my_nn
import read_image
import numpy as np
import struct
import math
import matplotlib.pyplot as plt

#基本测试，网上找了个case验证
x = 50 
print(x)
print(my_nn.Sigmoid.calValue(x))


inputvalue = [0] * 2
innervalue = [0] * 2
outputvalue = [0] * 1
layers = [my_nn.Layer(inputvalue), my_nn.Layer(innervalue), my_nn.Layer(outputvalue)]
weightsvalue1 = [[0.2, 0.8],[-0.7,-0.5]]
weightsvalue2 = [[0.3, 0.5]]
bvalue1 = [0,0]
bvalue2 = [0]
weights = [my_nn.Weight(weightsvalue1, bvalue1), my_nn.Weight(weightsvalue2, bvalue2)]
nn = my_nn.Neuralnetwork(layers,weights)


input_items = [0.3, -0.7]
output_items = [0.1]

print(input_items)
print(output_items)
nn.train(input_items, output_items, 1)
nn.printLayer(1)
nn.printLayer(2)
print()


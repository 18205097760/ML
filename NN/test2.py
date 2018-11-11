import my_nn
import read_image
import numpy as np
import struct
import math
import matplotlib.pyplot as plt

#自定义case验证，5个变量和是否大于2.5
#大于2.5（1,0） 否则 (0,1)
inputvalue = [0] * 5
innervalue = [0] * 3
outputvalue = [0] * 2
layers = [my_nn.Layer(inputvalue), my_nn.Layer(innervalue), my_nn.Layer(outputvalue)]
weightsvalue1 = np.random.rand(3, 5)
weightsvalue2 = np.random.rand(2, 3)
bvalue1 = np.random.rand(3)
bvalue2 = np.random.rand(2)
weights = [my_nn.Weight(weightsvalue1, bvalue1), my_nn.Weight(weightsvalue2, bvalue2)]
nn = my_nn.Neuralnetwork(layers,weights)

t = 10000
for i in range(t):
  input_items = np.random.rand(5)
  output_items = []
  a = 0
  b = 1
  if (input_items[0] + input_items[1] + input_items[2] + input_items[3] + input_items[4] > 2.5):
  	a = 1;
  	b = 0;

  output_items.append(a)
  output_items.append(b)
  print(input_items)
  print(input_items[0] + input_items[1] + input_items[2] + input_items[3] + input_items[4])
  nn.train(input_items, output_items, 1)
  nn.printLayer(2)
  print()

#验证学习结果
nn.calValue([0.1, 0.2, 0.3, 0.4, 0.5])
nn.printLayer(2)
nn.calValue([0.6, 0.2, 0.9, 0.4, 0.5])
nn.printLayer(2)
nn.calValue([0.7, 0.2, 0.3, 0.4, 0.5])
nn.printLayer(2)
nn.calValue([0.8, 0.7, 0.3, 0.4, 0.5])
nn.printLayer(2)

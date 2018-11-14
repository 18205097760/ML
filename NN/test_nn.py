import my_nn
import read_image
import numpy as np
import struct
import matplotlib.pyplot as plt

#28*28的数字识别，学习1000次，正确率72%（100个）
#28*28的数字识别，学习1000次，正确率63.8%（1000个）
#28*28的数字识别，学习1000次，正确率66.8%（1000个）
#28*28的数字识别，学习10000次，正确率89.4%（1000个）
#28*28的数字识别，学习60000次，正确率93%（1000个）

inputvalue = [0] * 784
innervalue = [0] * 49
outputvalue = [0] * 10
layers = [my_nn.Layer(inputvalue), my_nn.Layer(innervalue), my_nn.Layer(outputvalue)]
weightsvalue1 = np.random.rand(49, 784)
weightsvalue2 = np.random.rand(10, 49)
bvalue1 = np.random.rand(49)
bvalue2 = np.random.rand(10)
weights = [my_nn.Weight(weightsvalue1, bvalue1), my_nn.Weight(weightsvalue2, bvalue2)]
nn = my_nn.Neuralnetwork(layers,weights)

image_items = []
label_items = []
read_image.openImage('../data/MNIST/train-images.idx3-ubyte', '../data/MNIST/train-labels.idx1-ubyte', image_items, label_items)
print(len(image_items))
#训练次数
t = 6000
for i in range(t):
  output = [0] * 10
  for j in range(10):
    if (label_items[i] == j):
      output[j] = 1
  #print(image_items[i])
  inputimage = []
  for l in range(len(image_items[i])):
    inputimage.append(1.0 * image_items[i][l] / 255.0)
  nn.train(inputimage, output, 1)
  #print(output, label_items[i])
  #print(nn.printLayerMax(2), end=' ')
  #nn.printLayer(2)
  if (i % 100 == 0):
    print(i)

#验证次数
t = 1000
image_test_items = []
label_test_items = []
read_image.openImage('../data/MNIST/t10k-images.idx3-ubyte', '../data/MNIST/t10k-labels.idx1-ubyte', image_test_items, label_test_items)
print(len(image_test_items))
success = 0
fail = 0
for i in range(t):
  inputimage = []
  for l in range(len(image_test_items[i])):
    inputimage.append(1.0 * image_test_items[i][l] / 255.0)
  #print(inputimage)
  nn.calValue(inputimage)
  out = nn.printLayerMax(2)
  #print(label_test_items[i])
  print(nn.printLayerMax(2), end=' ')
  nn.printLayer(2)
  if (out == label_test_items[i]):
  	success += 1
  else:
  	fail += 1
print ("success: %d" % success)
print ("fail: %d" % fail)



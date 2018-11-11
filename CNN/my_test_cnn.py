import my_net
import my_func
import my_cnn
import my_image
import my_layer

import numpy as np
import struct
import matplotlib.pyplot as plt

# learn case:10000
# test:
#   success: 874
#   fail: 126
# [Finished in 426.8s]

func =  my_func.Sigmoid()
nn = my_net.Normal_NN([576, 10], func) #神经网络
cnn = my_cnn.Conv(28, [[5, 1]])  # 初始：28*28 卷积：7*7 步长：3

image_items = []
label_items = []
my_image.openImage('../data/MNIST/train-images.idx3-ubyte', '../data/MNIST/train-labels.idx1-ubyte', image_items, label_items)
print(len(image_items))

input_cnn = my_layer.TwoDimLayer(28)
error_cnn = my_layer.TwoDimLayer(24)
#训练次数
t = 10000
for i in range(t):
  output = [0] * 10
  for j in range(10):
    if (label_items[i] == j):
      output[j] = 1
  inputimage = []
  for l in range(len(image_items[i])):
    inputimage.append(1.0 * image_items[i][l] / 255.0)

  input_cnn.changeFromValue(inputimage)
  cnn_result = cnn.calValue(input_cnn.value)
  #print(cnn_result.value)

  nn.learn(cnn_result.changeOneValue(), output, 1)
  #print(nn.getErrors(0))
  error_cnn.changeFromValue(nn.getErrors(0))
  cnn.back_learn(error_cnn, 0.001)

  if (i % 100 == 0):
    print(i)
  
  #print("weight:")
  #cnn.printWeight(0)
#验证次数
t = 1000
image_test_items = []
label_test_items = []
my_image.openImage('../data/MNIST/t10k-images.idx3-ubyte', '../data/MNIST/t10k-labels.idx1-ubyte', image_test_items, label_test_items)
print(len(image_test_items))
success = 0
fail = 0
for i in range(t):
  inputimage = []
  for l in range(len(image_test_items[i])):
    inputimage.append(1.0 * image_test_items[i][l] / 255.0)

  input_cnn.changeFromValue(inputimage)
  cnn_result = cnn.calValue(input_cnn.value)
  #print(cnn_result.value)
  output = nn.calValue(cnn_result.changeOneValue())
  out = 0
  for j in range(10):
    if(output[j] > output[out]):
      out = j

  print('my-real:', out, label_test_items[i])
  print(output)
  print()
  if (out == label_test_items[i]):
    success += 1
  else:
    fail += 1
print ("success: %d" % success)
print ("fail: %d" % fail)



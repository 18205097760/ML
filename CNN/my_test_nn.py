import my_net
import my_func
import my_image

import numpy as np
import struct
import matplotlib.pyplot as plt

func =  my_func.Sigmoid()
nn = my_net.Normal_NN([784,10], func)

image_items = []
label_items = []
my_image.openImage('../data/MNIST/train-images.idx3-ubyte', '../data/MNIST/train-labels.idx1-ubyte', image_items, label_items)
print(len(image_items))
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
  nn.learn(inputimage, output, 1)
  if (i % 100 == 0):
    print(i)

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

  output = nn.calValue(inputimage)
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



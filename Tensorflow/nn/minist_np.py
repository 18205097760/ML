import numpy as np
import struct
import matplotlib.pyplot as plt

def openImage(image_file, label_file, image_items, label_items):
  #load image
  filename = image_file
  binfile = open(filename , 'rb')
  buf = binfile.read()
 
  index = 0
  magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
  index += struct.calcsize('>IIII')

  for i in range(numImages):
    im = struct.unpack_from('>784B' ,buf, index)
    index += struct.calcsize('>784B')
    im = np.array(im)
    image_items.append(im)

  #load label
  filename = label_file
  binfile = open(filename , 'rb')
  buf = binfile.read()
  index = 0
  magic, numLabels = struct.unpack_from('>II' , buf , index)
  index += struct.calcsize('>II')

  for i in range(numImages):
    label = struct.unpack_from('>1B' ,buf, index)
    index += struct.calcsize('>1B')
    label = np.array(label)
    label_items.append(label[0])

image_items = []
label_items = []
openImage('../../data/MNIST/train-images.idx3-ubyte', '../../data/MNIST/train-labels.idx1-ubyte', image_items, label_items)
pic_num = len(image_items)
train_num = 100
test_num = 1000

w1 = np.random.rand(784, 100) / 784.0
w2 = np.random.rand(100, 10) / 100.0
b1 = np.zeros((1, 100))
b2 = np.zeros((1, 10))
learning_rate = 0.1;

image_test_items = []
label_test_items = []
openImage('../../data/MNIST/t10k-images.idx3-ubyte', '../../data/MNIST/t10k-labels.idx1-ubyte', image_test_items, label_test_items)
pic_test_num = len(image_test_items)

hidden_func = 0

for i in range(10000):
    if (i % 1000 == 0):
        value = np.random.rand(1) * pic_test_num / test_num
        tran_set = int(value[0]) * test_num
        image_test_use = image_test_items[tran_set : tran_set + test_num]
        label_test_use = label_test_items[tran_set : tran_set + test_num]

        x = np.array(image_test_use).reshape(test_num,784) / 255.0
        test_y = np.array(label_test_use).reshape(test_num, 1)
        y_ = np.eye(10)[test_y].reshape(test_num, 10)
       
    else:
        value = np.random.rand(1) * pic_num / train_num
        tran_set = int(value[0]) * train_num
        image_use = image_items[tran_set : tran_set + train_num]
        label_use = label_items[tran_set : tran_set + train_num]

        x = np.array(image_use).reshape(train_num,784) / 255.0
        train_y = np.array(label_use).reshape(train_num, 1)
        y_ = np.eye(10)[train_y].reshape(train_num, 10)

    layer = np.dot(x, w1) + b1
    if hidden_func == 1:
        layer_act = 1.0/(1 + np.exp(-layer))
    else:
        layer_act = np.maximum(np.zeros(layer.shape), layer)

    y = np.dot(layer_act, w2) + b2
    y_sigmod = 1.0/(1 + np.exp(-y))

    #l = np.sum((y_ - y_sigmod) ** 2 / pic_num)
    l = np.sum(np.dot(y_.T, np.log(y_sigmod)) + np.dot((1 - y_).T, np.log(1 - y_sigmod)) / pic_num)

    if (i % 1000 == 0):
        print(l)
        real_num = np.argmax(y_, 1)
        pred_num = np.argmax(y_sigmod, 1)
        print(np.sum(pred_num == real_num))
        continue;

    #平方差收敛
    #dy_sigmod = y_sigmod - y_
    #dy = dy_sigmod * y_sigmod * (1.0 - y_sigmod)

    #交叉熵
    dy = y_sigmod - y_

    #print(dy_sigmod)  

    db2 = dy
    dw2 = np.dot(layer_act.T, dy)
    dlayer_act = np.dot(dy, w2.T) 

    if hidden_func == 1:
        dlayer = dlayer_act * layer_act * (1.0 - layer_act)
    else:
        dlayer = np.maximum(np.zeros(layer.shape), layer) / layer * dlayer_act

    db1 = dlayer
    dw1 = np.dot(x.T, dlayer)

    b2 -= learning_rate * db2.sum(axis=0) / train_num
    w2 -= learning_rate * dw2 / train_num
    b1 -= learning_rate * db1.sum(axis=0) / train_num
    w1 -= learning_rate * dw1 / train_num
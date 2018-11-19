import numpy as np
import struct
import matplotlib.pyplot as plt
import os, imageio


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

def test():
  image_items = []
  label_items = []
  openImage('../../data/MNIST/train-images.idx3-ubyte', '../../data/MNIST/train-labels.idx1-ubyte', image_items, label_items)

  show_image1 = np.array(image_items[0]).reshape(28, 28)
  show_image2 = np.array(image_items[1]).reshape(28, 28)
  show_image3 = np.array(image_items[2]).reshape(28, 28)
  image = np.zeros((28, 28, 3))

  for i in range(28):
    for j in range(28):
      image[i][j][0] = show_image1[i][j] / 255.0
      image[i][j][1] = show_image2[i][j] / 255.0
      image[i][j][2] = show_image3[i][j] / 255.0

  plt.figure()
  plt.imshow(image)
  plt.axis('off') # 不显示坐标轴
  plt.savefig('test.png')
  plt.show()

  imageio.mimsave(os.path.join('samples.gif'), [show_image1, show_image2, show_image3], fps=5)


#test()
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
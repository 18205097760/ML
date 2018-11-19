import image

def discriminator(image, reuse=None):
    with tf.variable_scope('discriminator', reuse = reuse):
        conv1 = tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same', \
        kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.Constant(0),activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(conv1, kernel_size=5, filters=128, strides=2, padding='same', \
        kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.Constant(0),activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(conv2, kernel_size=5, filters=256, strides=2, padding='same', \
        kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer=tf.Constant(0),activation=tf.nn.relu)

        net_input = tf.contrib.layers.flatten(conv3)
        net1 = tf.layers.dense(net1, units=10)
        net2 = tf.layers.dense(net2, units=1)


def generator(z, reust=None):
    with tf.variable_scope('generator', reuse = reuse):
        



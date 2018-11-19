from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def hidden_layer(layer_input, output_depth, scope='hidden_layer'):
    input_depth = layer_input.get_shape()[-1]
    print(scope)
    with tf.variable_scope(scope):
        w = tf.get_variable(initializer=tf.random_normal_initializer(stddev=0.1), shape=(input_depth, output_depth), name='weights')
        b = tf.get_variable(initializer=tf.constant_initializer(0), shape=(output_depth), name='bias')
        net = tf.matmul(layer_input, w) + b
        return net

def DNN(input, output_depths, scope='DNN'):
    net = input
    for i, output_depth in enumerate(output_depths):
        net = hidden_layer(net, output_depth, scope='layer%d' % i)
        net = tf.nn.relu(net)
    net = hidden_layer(net, 10, scope='classification')
    return net

minist = input_data.read_data_sets('./input_data',one_hot = True)

batch_size = 5
input_ph = tf.placeholder(dtype=tf.float32, shape=(None,784))
label_ph = tf.placeholder(dtype=tf.float32, shape=(None,10))

dnn = DNN(input_ph, [400, 200, 100])
loss = tf.losses.softmax_cross_entropy(logits=dnn, onehot_labels=label_ph)
#loss = tf.losses.mean_squared_error(dnn, label_ph)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

real_num = tf.argmax(label_ph, 1)
pred_num = tf.argmax(dnn, 1)
succ = tf.reduce_sum(tf.cast(tf.equal(real_num, pred_num), "float32"))

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for e in range(10000):
        image,labels =  minist.train.next_batch(batch_size)
        #print(sess.run(labels, feed_dict={input_ph:image, label_ph:labels}));
        sess.run(train_op, feed_dict={input_ph:image, label_ph:labels})
        if (e % 100 == 0):
            image2,labels2 =  minist.test.next_batch(100)
            #print(sess.run(real_num, feed_dict={input_ph:image, label_ph:labels}))
            #print(sess.run(pred_num, feed_dict={input_ph:image, label_ph:labels}))
            print(sess.run(succ, feed_dict={input_ph:image2, label_ph:labels2}))

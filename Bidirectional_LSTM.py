import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets("/tmp/data/",one_hot=True)

learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 256
n_classes = 10

x = tf.placeholder("float",[None,n_steps,n_input])
y = tf.placeholder("float",[None,n_classes])

weights = tf.Variable(tf.random_normal([n_hidden*2,n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BiBNN(x,weights,biases):
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,n_input])
    x = tf.split(x,n_steps)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias = 1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,
                                                            x,dtype = tf.float32)
    return  tf.matmul(outputs[-1],weights)+biases

pred = BiBNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step*batch_size < max_samples:
        batch_x,batch_y = minist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size,n_steps,n_input])
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Ite"+str(step*batch_size)+",Minibatch Loss="+format(loss)+", Accuracy"+format(acc))
        step +=1
    print ("Finish")

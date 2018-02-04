import numpy as np
import os
import tensorflow as tf
import pickle
import scipy


num_epoch = 20
batchSz = 128
num_char = 200

X = tf.placeholder(tf.float32, [None, 4096])
y = tf.placeholder(tf.float32, [None, num_char])
keep_prob = tf.placeholder(tf.float32)

# character dictionary
# f = open('char_dict','rb')
# char_dict = pickle.load(f) 
# char_set = []
# for i in range(num_char):
# char_set.append(list(char_dict.keys())[list(char_dict.values()).index(i)])

# Load data
f1 = open('trainX1','rb')
train_data_x = pickle.load(f1)
f1.close()
print "Load train_data_x Completed!"

f2=open('trainY1','rb')
train_data_y = pickle.load(f2)
f2.close()
print "Load train_data_y Completed!"

f3=open('testX1','rb')
test_data_x = pickle.load(f3)
f3.close()
print "Load test_data_x Completed!"

f4=open('testY1','rb')
test_data_y = pickle.load(f4)
f4.close()
print "Load test_data_y Completed!"

num_test = test_data_x.shape[0]

def get_batches(x, y, batchSz):
    n_batch = int(x.shape[0]/batchSz)
    for i in range(n_batch):
        batchX = x[i*batchSz : (i+1)*batchSz, :]
        batchY = y[i*batchSz : (i+1)*batchSz, :]
        yield batchX, batchY


def Network():
    x = tf.reshape(X, shape=[-1, 64, 64, 1])
    # 5 conv layers
    w_c1 = tf.get_variable(name='w_c1', shape=[3, 3, 1, 32], initializer=tf.random_normal_initializer(stddev=0.1))
    b_c1 = tf.get_variable(name='b_c1', shape=[32], initializer=tf.random_normal_initializer(stddev=0.1))
    w_c2 = tf.get_variable(name='w_c2', shape=[3, 3, 32, 64], initializer=tf.random_normal_initializer(stddev=0.1))
    b_c2 = tf.get_variable(name='b_c2', shape=[64], initializer=tf.random_normal_initializer(stddev=0.1))
    w_c3 = tf.get_variable(name='w_c3', shape=[3, 3, 64, 128], initializer=tf.random_normal_initializer(stddev=0.1))
    b_c3 = tf.get_variable(name='b_c3', shape=[128], initializer=tf.random_normal_initializer(stddev=0.1))
    w_c4 = tf.get_variable(name='w_c4', shape=[3, 3, 128, 256], initializer=tf.random_normal_initializer(stddev=0.1))
    b_c4 = tf.get_variable(name='b_c4', shape=[256], initializer=tf.random_normal_initializer(stddev=0.1))
    w_c5 = tf.get_variable(name='w_c5', shape=[3, 3, 256, 512], initializer=tf.random_normal_initializer(stddev=0.1))
    b_c5 = tf.get_variable(name='b_c5', shape=[512], initializer=tf.random_normal_initializer(stddev=0.1))


    conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1_2 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_2, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2_2 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3_2 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    conv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_2, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    conv4_2 = tf.nn.max_pool(conv4_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_2, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    conv5_2 = tf.nn.max_pool(conv5_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #conv5_3 = tf.nn.dropout(conv5_2, keep_prob)

    # fully connect layer
    w_d_1 = tf.Variable(tf.random_normal([2*2*512, 1024], stddev=0.1), name='w_d_1')
    b_d_1 = tf.Variable(tf.random_normal([1024], stddev=0.1), name='b_d_1')

    #w_d_2 = tf.Variable(tf.random_normal([1024, 1024], stddev=0.1), name='w_d_2')
    #b_d_2 = tf.Variable(tf.random_normal([1024], stddev=0.1), name='b_d_2')
    # Flatten
    dense = tf.reshape(conv5_2, [-1, w_d_1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d_1), b_d_1))
    #dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d_2), b_d_2))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(tf.random_normal([1024, num_char], stddev=0.1), name='w_out')
    b_out = tf.Variable(tf.random_normal([num_char], stddev=0.1), name='b_out')
    logits = tf.add(tf.matmul(dense, w_out), b_out)

    saver = tf.train.Saver()
    
    return logits, saver


def Train():
    logits, saver = Network()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        count = 0
        for e in range(num_epoch):
            for xs, ys in get_batches(train_data_x, train_data_y, batchSz):
                count += 1
                _, loss_ = sess.run([optimizer, loss], feed_dict={X:xs, y:ys, keep_prob:0.5})
                total_loss += loss_
                print "The average loss is %.3f" %(total_loss / count)
        rightCount = 0.
        for xs, ys in get_batches(test_data_x, test_data_y, batchSz):
            a = sess.run(accuracy, feed_dict={X: xs, y: ys, keep_prob: 1.})
            rightCount += a
        
        print "The accuracy is %.2f" %(rightCount/num_test)
        
        save_path = saver.save(sess, "/tmp/2_2conv.ckpt")
        print "Model saved in file: %s" % save_path


Train()
